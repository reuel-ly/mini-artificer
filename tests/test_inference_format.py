import json

from config import WEATHER_TOOL_SCHEMA, format_system_with_tools


def test_format_system_with_tools_uses_json_not_python_repr() -> None:
    content = format_system_with_tools(WEATHER_TOOL_SCHEMA)
    expected_json = json.dumps(WEATHER_TOOL_SCHEMA, indent=2)

    assert '"name": "get_weather"' in content
    assert "'name':" not in content
    assert content.endswith(expected_json)
    assert "call a function only when the request needs one" in content
    assert "plain text without a function call" in content
