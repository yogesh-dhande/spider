import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "provision_exp005_cloud_controller.py"
SPEC = importlib.util.spec_from_file_location("provision_exp005_cloud_controller", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_revision_is_resolved_before_controller_creation(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        MODULE,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or "a" * 40,
    )
    assert MODULE.resolve_revision("main") == "a" * 40
    assert calls == [
        (["git", "rev-parse", "--verify", "main^{commit}"], {"capture": True})
    ]


def test_startup_restores_state_and_always_stops() -> None:
    startup = MODULE.startup_script("a" * 40, Path("config.json"))
    assert "trap shutdown_controller EXIT" in startup
    assert "latest.tar.gz" in startup
    assert "git -C /opt/spider checkout -q " + "a" * 40 in startup
    assert "run_exp005_cloud_controller.py --config config.json" in startup
