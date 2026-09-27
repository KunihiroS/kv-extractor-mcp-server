"""Transport-level contract tests for the MCP stdio server.

MCP's stdio transport reserves stdout for JSON-RPC. These tests launch the real
entry point as a subprocess and assert that nothing but JSON-RPC ever reaches
stdout, in both logging modes, and that the server starts without an OpenAI key.
No network access or API key is needed: the LLM agents are created lazily on the
first tool call, and a tool call without a key must fail closed with a clear error.
"""
import importlib
import json
import os
import subprocess
import sys
import threading

import pytest

INIT = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
               "clientInfo": {"name": "pytest", "version": "0"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def _env_without_key(**extra):
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env.update(extra)
    return env


def _run_session(args, messages, env, timeout=60):
    """Start the server, send `messages`, close stdin, and collect stdout/stderr to EOF."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "kv_extractor_mcp_server", *args],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    stderr_lines = []
    t = threading.Thread(target=lambda: stderr_lines.extend(proc.stderr), daemon=True)
    t.start()
    for m in messages:
        proc.stdin.write(json.dumps(m) + "\n")
    proc.stdin.flush()
    # Read responses until every request id has been answered, then close stdin
    # so the server exits and any late buffered output is flushed and captured.
    expected = {m["id"] for m in messages if "id" in m}
    seen, stdout_lines = set(), []
    for line in proc.stdout:
        stdout_lines.append(line.rstrip("\n"))
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("id") in expected:
            seen.add(obj["id"])
        if seen == expected:
            proc.stdin.close()
    stdout_lines.extend(l.rstrip("\n") for l in proc.stdout)
    proc.wait(timeout=timeout)
    t.join(5)
    return proc.returncode, stdout_lines, stderr_lines


def _assert_stdout_is_jsonrpc(stdout_lines):
    non_json = []
    for line in stdout_lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            non_json.append(line)
            continue
        if not (isinstance(obj, dict) and obj.get("jsonrpc") == "2.0"):
            non_json.append(line)
    assert non_json == [], f"non JSON-RPC lines on stdout: {non_json}"


def test_help_exits_zero_without_api_key():
    proc = subprocess.run(
        [sys.executable, "-m", "kv_extractor_mcp_server", "--help"],
        capture_output=True, text=True, env=_env_without_key(), timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--log" in proc.stdout


@pytest.mark.parametrize("log_mode", ["off", "on"])
def test_stdout_carries_only_jsonrpc(tmp_path, log_mode):
    args = ["--log=off"] if log_mode == "off" else ["--log=on", f"--logfile={tmp_path / 'server.log'}"]
    rc, out, err = _run_session(args, [INIT, INITIALIZED, TOOLS_LIST], _env_without_key())
    _assert_stdout_is_jsonrpc(out)
    responses = {json.loads(l)["id"]: json.loads(l) for l in out if l.strip()}
    assert set(responses) == {1, 2}
    tools = {t["name"] for t in responses[2]["result"]["tools"]}
    assert tools == {"extract_json", "extract_yaml", "extract_toml"}
    if log_mode == "on":
        assert (tmp_path / "server.log").exists()


def test_tool_call_without_api_key_fails_closed():
    call = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "extract_json", "arguments": {"input_text": "Price: 100 JPY"}}}
    rc, out, err = _run_session(["--log=off"], [INIT, INITIALIZED, TOOLS_LIST, call], _env_without_key())
    _assert_stdout_is_jsonrpc(out)
    resp = next(json.loads(l) for l in out if l.strip() and json.loads(l).get("id") == 3)
    payload = resp["result"]["structuredContent"]
    assert payload["success"] is False
    assert "OPENAI_API_KEY" in payload["error"]
    assert "Traceback" not in json.dumps(resp)


def test_spacy_model_installer_targets_running_interpreter(monkeypatch):
    """The model installer must install into *this* interpreter and never touch stdout."""
    from kv_extractor_mcp_server import server

    monkeypatch.setattr(server.importlib.util, "find_spec", lambda name: None)  # no pip, like uvx
    monkeypatch.setattr(server.shutil, "which", lambda name: "/fake/bin/uv")
    # `spacy.cli.download` the *module* is shadowed by the `download` function on the package,
    # so resolve it explicitly instead of via a dotted string.
    dl = importlib.import_module("spacy.cli.download")
    monkeypatch.setattr(dl, "get_compatibility", lambda: {"en_core_web_sm": ["3.8.0"]})
    cmd = server._spacy_model_install_command("en_core_web_sm")
    assert cmd[:3] == ["/fake/bin/uv", "pip", "install"]
    assert cmd[3:5] == ["--python", sys.executable]
    assert cmd[-1].endswith("en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl")

    calls = {}

    class Done:
        returncode = 0
        stdout = "installer chatter\n"
        stderr = ""

    def fake_run(cmd, **kw):
        calls["cmd"], calls["kw"] = cmd, kw
        return Done()

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    server._install_spacy_model("en_core_web_sm")
    assert calls["kw"]["capture_output"] is True  # nothing inherits fd 1


def test_spacy_model_resolution_failure_does_not_exit(monkeypatch):
    from kv_extractor_mcp_server import server

    def boom():
        raise SystemExit(1)

    monkeypatch.setattr(importlib.import_module("spacy.cli.download"), "get_compatibility", boom)
    with pytest.raises(RuntimeError):
        server._spacy_model_install_command("en_core_web_sm")
