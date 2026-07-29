"""Tests for the HTTP layer, over a real socket.

The bug these cover: this is an HTTP/1.1 keep-alive server, and the browser's
fetch() always sends a JSON body -- even for /api/stop, which ignored it. The
unread bytes were then parsed as the next request line, so the first POST after
a Stop came back 501 Unsupported method and only the retry (on a fresh
connection) worked.

Nothing here starts the pipeline, so no microphone is opened: stop on an idle
pipeline is a no-op and persona switching only rewrites a prompt.
"""
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest

import config
import server
import voice_agent as va


@pytest.fixture(scope="module")
def live_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


@pytest.fixture
def conn(live_server):
    """One connection, reused across requests -- the browser does the same."""
    host, port = live_server
    c = http.client.HTTPConnection(host, port, timeout=10)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def restore_persona():
    before = va.active_persona, list(va.conversation)
    yield
    va.active_persona = before[0]
    va.conversation[:] = before[1]


def post(conn, path, body=None):
    payload = json.dumps({} if body is None else body).encode()
    conn.request("POST", path, body=payload,
                 headers={"Content-Type": "application/json",
                          "Content-Length": str(len(payload))})
    res = conn.getresponse()
    raw = res.read()
    return res.status, (json.loads(raw) if raw else {})


def get(conn, path):
    conn.request("GET", path)
    res = conn.getresponse()
    raw = res.read()
    return res.status, (json.loads(raw) if raw else {})


def test_keep_alive_is_in_use():
    """Without HTTP/1.1 the whole class of bug would not exist; pin it."""
    assert server.Handler.protocol_version == "HTTP/1.1"


def test_stop_then_another_post_on_the_same_connection(conn):
    """The reported bug: 501 on the request right after a Stop."""
    assert post(conn, "/api/stop")[0] == 200
    status, payload = post(conn, "/api/persona", {"persona": "jester"})
    assert status == 200, "a POST after Stop must not fail on the same connection"
    assert payload["persona"] == "jester"


def test_many_posts_in_a_row_on_one_connection(conn):
    """Stop/persona/stop/... must all succeed, not just every other one."""
    for i in range(6):
        route = "/api/stop" if i % 2 == 0 else "/api/persona"
        body = None if route == "/api/stop" else {"persona": "explorer"}
        status, _ = post(conn, route, body)
        assert status == 200, f"request {i} to {route} returned {status}"


def test_body_on_an_unknown_route_does_not_poison_the_connection(conn):
    assert post(conn, "/api/nope", {"junk": "x" * 200})[0] == 404
    assert post(conn, "/api/stop")[0] == 200


def test_get_after_post_on_the_same_connection(conn):
    assert post(conn, "/api/stop")[0] == 200
    status, payload = get(conn, "/api/status")
    assert status == 200
    assert payload["running"] is False


def test_post_without_a_body_still_works(conn):
    conn.request("POST", "/api/stop")
    res = conn.getresponse()
    res.read()
    assert res.status == 200


def test_malformed_json_is_tolerated(conn):
    conn.request("POST", "/api/persona", body=b"{not json",
                 headers={"Content-Type": "application/json",
                          "Content-Length": "9"})
    res = conn.getresponse()
    payload = json.loads(res.read())
    assert res.status == 200
    assert payload["persona"] == config.DEFAULT_PERSONA
    # The body was still consumed, so the connection stays usable.
    assert post(conn, "/api/stop")[0] == 200


def test_chunked_body_does_not_leave_the_socket_desynchronised(live_server):
    """A body we cannot drain must close the connection instead of misparsing."""
    host, port = live_server
    c = http.client.HTTPConnection(host, port, timeout=10)
    try:
        c.putrequest("POST", "/api/stop")
        c.putheader("Content-Type", "application/json")
        c.putheader("Transfer-Encoding", "chunked")
        c.endheaders()
        c.send(b"2\r\n{}\r\n0\r\n\r\n")
        res = c.getresponse()
        res.read()
        assert res.status == 200
        assert res.will_close, "connection should not be reused after a chunked body"
    finally:
        c.close()


def test_status_reports_the_active_persona(conn):
    post(conn, "/api/persona", {"persona": "cheerleader"})
    status, payload = get(conn, "/api/status")
    assert status == 200
    assert payload["persona"] == "cheerleader"
    assert payload["persona_name"] == config.PERSONAS["cheerleader"]["name"]


def test_personas_endpoint_lists_keys_the_ui_can_send(conn):
    status, payload = get(conn, "/api/personas")
    assert status == 200
    assert payload["default"] == config.DEFAULT_PERSONA
    assert {p["key"] for p in payload["personas"]} == set(config.PERSONAS)
    for p in payload["personas"]:
        assert p["name"] and p["blurb"]
