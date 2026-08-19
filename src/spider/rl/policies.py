from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from spider.rl.types import Observation, PolicyTask, TaskSpec


class QwenActionPolicy:
    """Direct Qwen vision policy using EXP004's rationale-plus-action text protocol."""

    def __init__(self, *, policy_id: str, config_path: str, adapter: str | None) -> None:
        from spider.config import load_config
        from spider.evaluate import load_model

        self.policy_id = policy_id
        self.config = load_config(config_path)
        self.model, self.processor = load_model(self.config["experiment"], adapter)
        self.task: PolicyTask | None = None
        self.history: list[dict[str, object]] = []
        self.last_raw_output: str | None = None

    def start_episode(self, task: PolicyTask, seed: int) -> None:
        del seed
        self.task = task
        self.history = []
        self.last_raw_output = None

    def predict(self, observation: Observation) -> str:
        from PIL import Image

        from spider.prompts import action_prompt, inference_messages
        from spider.web_actions import WebActionError, parse_action_response, to_rollout_action

        if self.task is None:
            raise RuntimeError("Qwen policy must start an episode before prediction")
        image = Image.open(io.BytesIO(observation.screenshot)).convert("RGB")
        prompt = action_prompt(
            self.task.instruction,
            self.history[-int(self.config["data"]["max_past_steps"]) :],
            page_url=observation.url,
        )
        # The manifest evaluator loads images from disk; live rollouts use the same prompt and
        # generation protocol directly from the in-memory screenshot.
        messages = inference_messages("action", prompt, image)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **self.config["experiment"].get("chat_template_kwargs", {}),
        ).to(self.model.device)
        import torch

        from spider.evaluate import generation_eos_token_ids

        kwargs: dict[str, object] = {
            "max_new_tokens": int(self.config["evaluation"]["max_new_tokens_action"]),
            "do_sample": False,
            "use_cache": True,
        }
        eos_ids = generation_eos_token_ids(self.model, self.processor)
        if eos_ids:
            kwargs["eos_token_id"] = eos_ids
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        raw = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        self.last_raw_output = raw
        try:
            parsed = parse_action_response(raw)
            action = to_rollout_action(parsed)
        except WebActionError:
            # Let the rollout runner record a parse error and the exact model response instead of
            # aborting the complete study.
            return raw
        self.history.append(
            {
                "index": observation.step_index,
                "thought": parsed["thought"],
                "action": parsed["action"],
            }
        )
        return json.dumps(action.to_dict(), sort_keys=True, separators=(",", ":"))

    def close(self) -> None:
        import gc

        import torch

        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OraclePolicy:
    """Deterministic test policy used to validate rollout and ablation infrastructure."""

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        policy_id: str,
        coordinate_bias_px: tuple[float, float] = (0.0, 0.0),
        malformed_every: int = 0,
    ) -> None:
        self.policy_id = policy_id
        self.tasks = {task.task_id: task for task in tasks}
        self.coordinate_bias_px = coordinate_bias_px
        self.malformed_every = malformed_every
        self.episode_prediction_count = 0

    def start_episode(self, task: PolicyTask, seed: int) -> None:
        del task, seed
        self.episode_prediction_count = 0

    def predict(self, observation: Observation) -> str:
        self.episode_prediction_count += 1
        if (
            self.malformed_every
            and self.episode_prediction_count % self.malformed_every == 0
        ):
            return "not-json"
        task = self.tasks[observation.task_id]
        expected = task.steps[observation.step_index]
        payload: dict[str, object] = {"action": expected.action}
        if expected.action == "click":
            target = next(item for item in task.elements if item.element_id == expected.target)
            left, top, right, bottom = target.bbox
            bias_x, bias_y = self.coordinate_bias_px
            payload["x"] = ((left + right) / 2 + bias_x) / task.viewport[0]
            payload["y"] = ((top + bottom) / 2 + bias_y) / task.viewport[1]
        elif expected.action == "type":
            payload["text"] = expected.text
        elif expected.action == "scroll":
            payload["direction"] = expected.direction
            payload["amount"] = expected.amount
        elif expected.action == "done":
            payload["result"] = expected.result or ""
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class HttpPolicy:
    """Platform-neutral policy client for local or managed GPU model servers."""

    def __init__(
        self,
        *,
        policy_id: str,
        endpoint: str,
        timeout_seconds: float = 120.0,
        bearer_token: str | None = None,
        max_history: int = 4,
    ) -> None:
        if urlsplit(endpoint).scheme not in {"http", "https"}:
            raise ValueError("HTTP policy endpoint must use http or https")
        self.policy_id = policy_id
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.bearer_token = bearer_token
        self.max_history = max_history
        self.task: PolicyTask | None = None
        self.seed = 0
        self.history: list[str] = []

    def start_episode(self, task: PolicyTask, seed: int) -> None:
        self.task = task
        self.seed = seed
        self.history = []

    def predict(self, observation: Observation) -> str:
        if self.task is None:
            raise RuntimeError("HTTP policy must start an episode before prediction")
        payload = {
            "schema_version": 1,
            "policy_id": self.policy_id,
            "task": {
                "task_id": self.task.task_id,
                "instruction": self.task.instruction,
                "seed": self.seed,
            },
            "observation": {
                "task_id": observation.task_id,
                "step_index": observation.step_index,
                "url": observation.url,
                "width": observation.width,
                "height": observation.height,
                "screenshot_base64": base64.b64encode(observation.screenshot).decode("ascii"),
            },
            "previous_outputs": self.history[-self.max_history :],
        }
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Policy request failed for {self.policy_id}: {error}") from error
        if not isinstance(response_payload, dict):
            raise TypeError("Policy response must be a JSON object")
        if isinstance(response_payload.get("output"), str):
            output = response_payload["output"]
        elif isinstance(response_payload.get("action"), dict):
            output = json.dumps(
                response_payload["action"], sort_keys=True, separators=(",", ":")
            )
        else:
            raise TypeError("Policy response requires string output or object action")
        self.history.append(output)
        return output


def make_policy(config: dict[str, object], tasks: Sequence[TaskSpec]):
    policy_type = config.get("type")
    policy_id = config.get("id")
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError("policy.id must be a non-empty string")
    if policy_type == "oracle":
        bias = config.get("coordinate_bias_px", [0.0, 0.0])
        if not isinstance(bias, list) or len(bias) != 2:
            raise ValueError("policy.coordinate_bias_px must contain two numbers")
        return OraclePolicy(
            tasks,
            policy_id=policy_id,
            coordinate_bias_px=(float(bias[0]), float(bias[1])),
            malformed_every=int(config.get("malformed_every", 0)),
        )
    if policy_type == "http":
        endpoint_env = str(config.get("endpoint_env", "SPIDER_POLICY_ENDPOINT"))
        endpoint = os.environ.get(endpoint_env)
        if not endpoint:
            raise ValueError(f"HTTP policy endpoint environment variable is unset: {endpoint_env}")
        token_env = config.get("bearer_token_env")
        token = os.environ.get(str(token_env)) if token_env else None
        return HttpPolicy(
            policy_id=policy_id,
            endpoint=endpoint,
            timeout_seconds=float(config.get("timeout_seconds", 120.0)),
            bearer_token=token,
            max_history=int(config.get("max_history", 4)),
        )
    if policy_type == "qwen":
        adapter_env = str(config.get("adapter_env", "SPIDER_POLICY_ADAPTER"))
        adapter = os.environ.get(adapter_env) or config.get("adapter_path")
        return QwenActionPolicy(
            policy_id=policy_id,
            config_path=str(config.get("config_path", "configs/experiment4.yaml")),
            adapter=str(adapter) if adapter else None,
        )
    raise ValueError(f"Unsupported policy type: {policy_type!r}")
