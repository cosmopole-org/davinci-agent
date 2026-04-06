import json

from agentic_runtime import capability_report


def test_capability_report_shape():
    report = capability_report()

    assert "generated_at_utc" in report
    assert "supported_count" in report
    assert "unsupported_count" in report
    assert isinstance(report["features"], list)
    assert isinstance(report["vm_tool_coverage"], dict)
    assert report["supported_count"] + report["unsupported_count"] == len(report["features"])


def test_capability_report_json_serializable():
    report = capability_report()
    payload = json.dumps(report)
    assert isinstance(payload, str)
    assert payload.startswith("{")
