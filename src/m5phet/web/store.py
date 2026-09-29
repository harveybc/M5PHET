"""Single-owner chat persistence, separate from scientific/governed evidence."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

from .engine import DEFAULT_CONFIG, validate_config

#: AP01: one sentence for one event, wherever it is recorded from -- the run thread that saw the shutdown
#: (`web.app.interrupted_or_refused`) or the next start-up finding a row nobody got to record (`Store.recover`).
STOPPED_WHILE_RUNNING = ("Server stopped while this request was running; no answer was produced. Send the same "
                         "request again to run it.")


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Store:
    def __init__(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = root / "chat.sqlite3"
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS chats(id TEXT PRIMARY KEY,title TEXT NOT NULL,config TEXT NOT NULL,created TEXT,updated TEXT);
            CREATE TABLE IF NOT EXISTS files(id TEXT PRIMARY KEY,chat TEXT REFERENCES chats(id) ON DELETE CASCADE,
              name TEXT,sha TEXT,data BLOB,created TEXT);
            CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,chat TEXT REFERENCES chats(id) ON DELETE CASCADE,
              client_id TEXT,request_digest TEXT,role TEXT,content TEXT,status TEXT,detail TEXT,created TEXT,
              UNIQUE(chat,client_id,role));
            """)
        os.chmod(self.path, 0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA secure_delete=ON")
        return db

    def recover(self):
        """Run at every start-up: a request that was in flight when the process went away is named, never guessed.

        AP01: the message says the request may be sent again, because since this round it can be -- `begin` re-runs an
        INTERRUPTED request of the same identity instead of handing back a record that would never complete."""
        with self.connect() as db:
            db.execute("UPDATE messages SET status='INTERRUPTED',content=? WHERE status='RUNNING'",
                       (STOPPED_WHILE_RUNNING,))

    def interrupt_stopped(self, ids, shadowed=None):
        """Relabel the requests that were in flight when the server began stopping and then failed.

        AP01, and the reason the shutdown flag alone is not enough: stopping the process kills the engines'
        subprocesses at the same instant, so the run thread's exception frequently arrives BEFORE the flag is set
        (measured 2026-09-29: `PROVIDER_ERROR: native CPU forecast process refused:` recorded on a request that was
        merely running when the server went down). This runs after the pool is drained, over the ids that were in
        flight when the stop began, and it is deliberately narrow:

        * an answer that completed is left exactly as it is -- a shutdown never rewrites an answer;
        * a TYPED refusal is left too: it carries a `response`, which means an engine considered the question and
          declined it by name, and that is the person's answer;
        * a run that ended in an exception with no response at all is renamed, because its recorded cause is the
          shutdown wearing an engine's name;
        * and `shadowed(detail)` -- supplied by the caller, which owns the meaning of an engine's answers -- renames
          the case where the engines DID return, all of them saying only that their own process failed.

        The engine's own words are kept in `error_at_shutdown`: evidence is never discarded, it is just not
        presented as the answer."""
        with self.connect() as db:
            for mid in list(ids or ()):
                row = db.execute("SELECT status,content,detail FROM messages WHERE id=?", (mid,)).fetchone()
                if row is None or row["status"] in ("OK", "PARTIAL", "INTERRUPTED"):
                    continue
                detail = json.loads(row["detail"])
                if "response" in detail and not (shadowed and shadowed(detail)):
                    continue
                detail = detail | {"stopped_while_running": True, "error_at_shutdown": row["content"]}
                db.execute("UPDATE messages SET status='INTERRUPTED',content=?,detail=? WHERE id=?",
                           (STOPPED_WHILE_RUNNING, dump(detail), mid))

    def create(self, title):
        cid, clock = uuid.uuid4().hex, now()
        with self.connect() as db:
            db.execute("INSERT INTO chats VALUES(?,?,?,?,?)", (cid, title, dump(DEFAULT_CONFIG), clock, clock))
        return self.get(cid)

    def list(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT id,title,created,updated FROM chats ORDER BY updated DESC")]

    def get(self, cid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise KeyError("Chat not found")
            result = dict(row)
            result["config"] = json.loads(result["config"])
            result["messages"] = [{**dict(r), "detail": json.loads(r["detail"])} for r in
                                  db.execute("SELECT * FROM messages WHERE chat=? ORDER BY rowid", (cid,))]
            result["files"] = [{"id": r["id"], "name": r["name"], "sha256": r["sha"], "size": r["size"]} for r in
                               db.execute("SELECT id,name,sha,length(data) AS size FROM files WHERE chat=? ORDER BY rowid", (cid,))]
        return result

    def update(self, cid, title=None, config=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise KeyError("Chat not found")
            db.execute("UPDATE chats SET title=?,config=?,updated=? WHERE id=?",
                       (title if title is not None else row["title"], dump(validate_config(config)) if config is not None else row["config"], now(), cid))
        return self.get(cid)

    def delete(self, cid):
        self.get(cid)
        with self.connect() as db:
            if db.execute("SELECT 1 FROM messages WHERE chat=? AND status='RUNNING'", (cid,)).fetchone():
                raise RuntimeError("Wait for the active request before deleting this chat")
            db.execute("DELETE FROM chats WHERE id=?", (cid,))
        # SQLite reuses freed pages. secure_delete prevents deleted upload bytes remaining in those pages.

    def upload(self, cid, name, data):
        fid, sha = uuid.uuid4().hex, hashlib.sha256(data).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM chats WHERE id=?", (cid,)).fetchone():
                raise KeyError("Chat not found")
            size = db.execute("SELECT coalesce(sum(length(data)),0) FROM files").fetchone()[0]
            if size + len(data) > 250 * 1024 * 1024:
                raise ValueError("Local attachment quota is 250 MiB")
            db.execute("INSERT INTO files VALUES(?,?,?,?,?,?)", (fid, cid, name, sha, data, now()))
        return {"id": fid, "name": name, "sha256": sha, "size": len(data)}

    def file(self, cid, fid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM files WHERE chat=? AND id=?", (cid, fid)).fetchone()
        if row is None:
            raise KeyError("Attachment not found in this chat")
        row = dict(row)
        if hashlib.sha256(row["data"]).hexdigest() != row["sha"]:
            raise ValueError("Attachment failed integrity validation")
        return row

    def begin(self, cid, client_id, prompt, file_ids, extra=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            chat = db.execute("SELECT * FROM chats WHERE id=?", (cid,)).fetchone()
            if chat is None:
                raise KeyError("Chat not found")
            # an envelope the person edited is a different request: it is part of the identity, not a detail of it
            fingerprint = hashlib.sha256(dump({"prompt": prompt, "files": file_ids, "extra": extra}).encode()).hexdigest()
            prior = db.execute("SELECT * FROM messages WHERE chat=? AND client_id=? AND role='assistant'", (cid, client_id)).fetchone()
            if prior:
                if prior["request_digest"] != fingerprint:
                    raise RuntimeError("Request ID was already used for different input")
                if prior["status"] != "INTERRUPTED":
                    return prior["id"], None
                # AP01: a request the restart interrupted has no answer, so re-sending the same request id cannot be
                # a replay of one. Before this, `begin` handed the interrupted record back and ran nothing: the
                # caller got 202 for work that would never start and polled a message that would never finish, and
                # the only way out was to invent a new request id -- which is a DIFFERENT request. It is run once,
                # from the snapshot it was accepted with, so what runs is the request the person sent and not
                # whatever the chat's configuration happens to be now.
                if db.execute("SELECT 1 FROM messages WHERE status='RUNNING'").fetchone():
                    raise RuntimeError("Another request is running; try again when it finishes")
                resumed = json.loads(prior["detail"])
                again = [self.file(cid, item["id"]) for item in resumed.get("attachments", [])]
                db.execute("UPDATE messages SET status='RUNNING',content='' WHERE id=?", (prior["id"],))
                db.execute("UPDATE chats SET updated=? WHERE id=?", (now(), cid))
                return prior["id"], (prompt, resumed["config"], again, resumed)
            if db.execute("SELECT 1 FROM messages WHERE status='RUNNING'").fetchone():
                raise RuntimeError("Another request is running; try again when it finishes")
            attachments = [self.file(cid, fid) for fid in file_ids]
            config, mid, clock = json.loads(chat["config"]), uuid.uuid4().hex, now()
            snapshot = {"config": config, "attachments": [{"id": f["id"], "name": f["name"], "sha256": f["sha"]} for f in attachments]}
            if extra is not None:
                snapshot["envelope"] = extra
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                       (uuid.uuid4().hex,cid,client_id,fingerprint,"user",prompt,"SENT",dump(snapshot),clock))
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                       (mid,cid,client_id,fingerprint,"assistant","","RUNNING",dump(snapshot),clock))
            db.execute("UPDATE chats SET updated=? WHERE id=?", (clock, cid))
        return mid, (prompt, config, attachments, snapshot)

    def finish(self, mid, status, content, detail):
        with self.connect() as db:
            db.execute("UPDATE messages SET status=?,content=?,detail=? WHERE id=?", (status,content,dump(detail),mid))
