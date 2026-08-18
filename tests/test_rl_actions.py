import pytest

from spider.rl.actions import ActionParseError, BrowserAction, parse_action


def test_parse_browser_actions() -> None:
    assert parse_action('{"action":"click","x":0.25,"y":0.75}') == BrowserAction(
        action="click", x=0.25, y=0.75
    )
    assert parse_action('{"action":"type","text":"hello"}').text == "hello"
    assert parse_action('{"action":"scroll","direction":"down","amount":0.5}').amount == 0.5
    assert parse_action('{"action":"go_back"}').action == "go_back"
    assert parse_action('{"action":"done","result":"ok"}').result == "ok"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '[]',
        '{"action":"click","x":2,"y":0}',
        '{"action":"click","x":0.5,"y":0.5,"text":"extra"}',
        '{"action":"scroll","direction":"left","amount":0.5}',
        '{"action":"type","text":""}',
    ],
)
def test_reject_invalid_browser_actions(raw: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(raw)
