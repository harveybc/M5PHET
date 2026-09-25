"""WP12: a program's bearer token beside the owner's cookie, and what must never happen to it.

The workbench is the owner's browser session, authenticated by a cookie the login dialog sets. A program has no
browser, and giving it the owner's token to post at `/api/login` means putting that token in a script. So the API
takes a SECOND credential: `Authorization: Bearer <token>`, where the token is the contents of a file the operator
names -- never a value written in a configuration file, never an argument in a command line.

The rules tested here, each of which is a refusal and not a repair:

* the right token is accepted on `/api/*`, and may also open a session at `/api/login`;
* a wrong token is 401, and so is a bearer token when no file names one -- the cookie is then the only way in;
* the host allow-list and the cross-origin rules apply to a bearer request exactly as they do to a browser's;
* the token value never appears in a response body, a response header, or a log line. A credential that is echoed
  back in an error message, or written to a log the owner later pastes into a report, has stopped being a secret.
"""
import logging

import pytest
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from m5phet.config import Configuration
from m5phet.runtime import Registry
from m5phet.web.app import API_TOKEN_VARIABLE, api_token_path, create_app, read_api_token
from m5phet.web.engine import Engine

OWNER = "owner-access-token-for-tests"
TOKEN = "bearer-token-3f9a2c7e51d84b06"


def token_file(tmp_path, value=TOKEN, name="api.token"):
    path = tmp_path / name
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def instance(tmp_path, *, token_path=None, access_token=OWNER, state="state"):
    return create_app(tmp_path / state, engine=Engine(registry=Registry()), access_token=access_token,
                      api_token_file=token_path)


def client(app, host="127.0.0.1"):
    return TestClient(app, base_url=f"http://{host}")


def bearer(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


# --- the token is accepted -------------------------------------------------------------------------------------------

def test_bearer_token_is_accepted_on_the_api(tmp_path):
    with client(instance(tmp_path, token_path=token_file(tmp_path))) as c:
        assert c.get("/api/chats").status_code == 401, "no credential must still be refused"
        assert c.get("/api/chats", headers=bearer()).status_code == 200
        assert c.get("/api/catalog", headers=bearer()).status_code == 200
        created = c.post("/api/chats", json={"title": "from a program"}, headers=bearer())
        assert created.status_code == 201, created.text
        cid = created.json()["id"]
        assert c.get(f"/api/chats/{cid}", headers=bearer()).json()["title"] == "from a program"
        assert c.get("/api/tasks/catalog", headers=bearer()).json()["execution_authorized"] is False


def test_bearer_token_opens_a_session_and_the_cookie_then_carries_it(tmp_path):
    """A client may keep the cookie instead of resending the token on every call; both are the same access."""
    with client(instance(tmp_path, token_path=token_file(tmp_path))) as c:
        assert c.post("/api/login", json={}, headers=bearer()).status_code == 200
        assert c.get("/api/chats").status_code == 200, "the login cookie is now carried by the client"


def test_the_owner_token_still_works_and_a_trailing_newline_in_the_file_is_not_part_of_the_token(tmp_path):
    path = tmp_path / "api.token"
    path.write_text(f"  {TOKEN}\n\n", encoding="utf-8")
    with client(instance(tmp_path, token_path=path)) as c:
        assert c.get("/api/chats", headers=bearer()).status_code == 200
        assert c.post("/api/login", json={"token": OWNER}).status_code == 200
        assert c.get("/api/chats").status_code == 200


# --- the token is refused --------------------------------------------------------------------------------------------

def test_a_wrong_token_is_401_and_changes_nothing(tmp_path):
    with client(instance(tmp_path, token_path=token_file(tmp_path))) as c:
        for presented in (TOKEN[:-1], TOKEN + "x", TOKEN.upper(), "", OWNER, "null"):
            assert c.get("/api/chats", headers=bearer(presented)).status_code == 401, presented
            assert c.post("/api/chats", json={"title": "no"}, headers=bearer(presented)).status_code == 401
        assert c.get("/api/chats", headers={"Authorization": TOKEN}).status_code == 401, "no scheme, no access"
        assert c.get("/api/chats", headers={"Authorization": f"Basic {TOKEN}"}).status_code == 401
        assert c.post("/api/login", json={}, headers=bearer("wrong")).status_code == 401
        assert c.get("/api/chats", headers=bearer()).json() == [], "nothing was created by the refused calls"


def test_without_a_token_file_the_bearer_header_is_ignored_and_the_cookie_is_still_required(tmp_path):
    app = instance(tmp_path, token_path=tmp_path / "there-is-no-such-file")
    with client(app) as c:
        for presented in (TOKEN, OWNER, "anything"):
            assert c.get("/api/chats", headers=bearer(presented)).status_code == 401, presented
        assert c.post("/api/login", json={}, headers=bearer()).status_code == 401
        assert c.post("/api/login", json={"token": OWNER}).status_code == 200
        assert c.get("/api/chats").status_code == 200


def test_an_empty_or_unreadable_token_file_leaves_bearer_access_off(tmp_path):
    assert read_api_token(token_file(tmp_path, value="   ", name="blank.token")) is None
    assert read_api_token(tmp_path / "absent") is None
    assert read_api_token(tmp_path) is None, "a directory is not a token"
    assert read_api_token(None) is None
    with client(instance(tmp_path, token_path=token_file(tmp_path, value="", name="empty.token"))) as c:
        assert c.get("/api/chats", headers=bearer("")).status_code == 401
        assert c.get("/api/chats", headers={"Authorization": "Bearer"}).status_code == 401


def test_the_host_allow_list_and_the_origin_rule_apply_to_a_bearer_request(tmp_path):
    app = create_app(tmp_path / "remote", engine=Engine(registry=Registry()), access_token=OWNER,
                     allowed_hosts=["chat.allowed"], api_token_file=token_file(tmp_path))
    with client(app, host="chat.allowed") as c:
        assert c.get("/api/chats", headers=bearer()).status_code == 200
        assert c.get("/api/chats", headers=bearer() | {"host": "elsewhere.example"}).status_code == 403
        assert c.get("/api/chats", headers=bearer() | {"origin": "https://elsewhere.example"}).status_code == 403
        assert c.get("/api/chats", headers=bearer() | {"sec-fetch-site": "cross-site"}).status_code == 403


# --- where the file is named ------------------------------------------------------------------------------------------

def test_the_variable_names_the_file_and_wins_over_the_configuration(tmp_path):
    bound = Configuration(data={"surfaces": {"api": {"token_file": str(tmp_path / "from-json.token")}}})
    assert api_token_path(bound, environ={}) == tmp_path / "from-json.token"
    assert api_token_path(bound, environ={API_TOKEN_VARIABLE: str(tmp_path / "from-env.token")}) == \
        tmp_path / "from-env.token"
    assert api_token_path(None, environ={}) is None, "neither names one: bearer access is off"
    assert api_token_path(Configuration(data={"surfaces": {"web": {"port": 8765}}}), environ={}) is None


def test_the_variable_turns_bearer_access_on_for_an_instance_that_builds_its_own_engine(tmp_path, monkeypatch):
    path = token_file(tmp_path)
    monkeypatch.setenv(API_TOKEN_VARIABLE, str(path))
    with client(create_app(tmp_path / "by-variable", engine=Engine(registry=Registry()), access_token=OWNER)) as c:
        assert c.get("/api/chats", headers=bearer()).status_code == 200
        assert c.get("/api/chats", headers=bearer("wrong")).status_code == 401


def test_a_token_file_that_appears_after_start_up_does_not_turn_access_on(tmp_path):
    """The rule is read once. Access that changes because a file was written while the server runs is not a rule."""
    later = tmp_path / "later.token"
    with client(instance(tmp_path, token_path=later)) as c:
        later.write_text(TOKEN, encoding="utf-8")
        assert c.get("/api/chats", headers=bearer()).status_code == 401


# --- the token never leaves ---------------------------------------------------------------------------------------

def test_the_token_value_never_appears_in_a_response_or_a_log_line(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    app = instance(tmp_path, token_path=token_file(tmp_path))
    seen = []
    with client(app) as c:
        cid = c.post("/api/chats", json={"title": "t"}, headers=bearer()).json()["id"]
        seen.append(c.post("/api/login", json={}, headers=bearer()))
        seen.append(c.get("/api/catalog", headers=bearer()))
        seen.append(c.get("/api/tasks/catalog", headers=bearer()))
        seen.append(c.get("/api/chats", headers=bearer()))
        seen.append(c.get(f"/api/chats/{cid}", headers=bearer()))
        seen.append(c.get(f"/api/chats/{cid}/export", headers=bearer()))
        seen.append(c.post(f"/api/chats/{cid}/files", files={"file": ("d.csv", b"a,b\n1,2\n", "text/csv")},
                           headers=bearer()))
        # the refusals are where a credential usually leaks: an error message that quotes what was sent
        seen.append(c.get("/api/chats", headers=bearer("wrong-" + TOKEN)))
        seen.append(c.get("/api/chats/does-not-exist", headers=bearer()))
        seen.append(c.patch(f"/api/chats/{cid}", json={"unknown": 1}, headers=bearer()))
        seen.append(c.post(f"/api/chats/{cid}/messages", json={"client_id": "c"}, headers=bearer()))
        seen.append(c.get("/api/chats", headers=bearer() | {"host": "elsewhere.example"}))
    for response in seen:
        assert TOKEN not in response.text, response.request.url
        assert TOKEN not in str(dict(response.headers))
        assert TOKEN not in str(dict(response.cookies))
    assert TOKEN not in caplog.text
    assert TOKEN not in (tmp_path / "state" / "chat.sqlite3").read_bytes().decode("latin-1")


def test_the_catalog_does_not_publish_the_token_nor_its_contents(tmp_path):
    path = token_file(tmp_path)
    bound = Configuration(data={"surfaces": {"api": {"token_file": str(path)}}})
    app = create_app(tmp_path / "published", engine=Engine(registry=Registry(), configuration=bound),
                     access_token=OWNER)
    with client(app) as c:
        published = c.get("/api/catalog", headers=bearer()).text
        assert TOKEN not in published
        assert '"api"' in published, "the surface is declared; only its contents are not"
