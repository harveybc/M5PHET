"""Single-owner chat persistence, separate from scientific/governed evidence."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

from .engine import DEFAULT_CONFIG, validate_config


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
        with self.connect() as db:
            db.execute("UPDATE messages SET status='INTERRUPTED',content='Server restarted before completion; not retried automatically.' WHERE status='RUNNING'")

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
                return prior["id"], None
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
