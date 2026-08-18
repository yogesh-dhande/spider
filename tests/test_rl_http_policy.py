import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spider.rl.policies import HttpPolicy, make_policy
from spider.rl.types import Observation, PolicyTask


class _PolicyHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.received.append(json.loads(self.rfile.read(length)))
        body = json.dumps({"action": {"action": "click", "x": 0.25, "y": 0.75}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        del format, args


def test_http_policy_request_and_history() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PolicyHandler)
    server.received = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = HttpPolicy(
            policy_id="qwen-test",
            endpoint=f"http://127.0.0.1:{server.server_port}/predict",
        )
        policy.start_episode(PolicyTask(task_id="task", instruction="Click it"), seed=17)
        observation = Observation(
            task_id="task",
            instruction="Click it",
            step_index=0,
            screenshot=b"png-bytes",
            width=640,
            height=360,
            url="https://sandbox.local",
        )
        assert json.loads(policy.predict(observation)) == {
            "action": "click",
            "x": 0.25,
            "y": 0.75,
        }
        policy.predict(observation)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    first, second = server.received
    assert base64.b64decode(first["observation"]["screenshot_base64"]) == b"png-bytes"
    assert first["previous_outputs"] == []
    assert len(second["previous_outputs"]) == 1


def test_http_policy_factory_reads_endpoint_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("TEST_POLICY_ENDPOINT", "http://127.0.0.1:1/predict")
    policy = make_policy(
        {"type": "http", "id": "remote-qwen", "endpoint_env": "TEST_POLICY_ENDPOINT"},
        [],
    )
    assert isinstance(policy, HttpPolicy)
    assert policy.endpoint == "http://127.0.0.1:1/predict"
