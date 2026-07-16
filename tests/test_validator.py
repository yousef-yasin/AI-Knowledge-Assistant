from types import SimpleNamespace

from src import validator


def test_validate_answer_parses_valid_json(monkeypatch) -> None:
    response_text = (
        '{"status": "PASS", "reason": "Supported", "supported": true}'
    )

    monkeypatch.setattr(
        validator.client.models,
        "generate_content",
        lambda **kwargs: SimpleNamespace(text=response_text),
    )

    result = validator.validate_answer(
        question="How many leave days?",
        context="Employees receive 21 leave days.",
        answer="Employees receive 21 leave days.",
    )

    assert result["status"] == "PASS"
    assert result["supported"] is True


def test_validate_answer_accepts_json_code_fence(monkeypatch) -> None:
    response_text = (
        '```json\n'
        '{"status": "PASS", "reason": "Supported", "supported": true}'
        '\n```'
    )

    monkeypatch.setattr(
        validator.client.models,
        "generate_content",
        lambda **kwargs: SimpleNamespace(text=response_text),
    )

    result = validator.validate_answer("Q", "Context", "Answer")

    assert result["status"] == "PASS"
    assert result["supported"] is True


def test_validate_answer_handles_invalid_response(monkeypatch) -> None:
    monkeypatch.setattr(
        validator.client.models,
        "generate_content",
        lambda **kwargs: SimpleNamespace(text="not json"),
    )

    result = validator.validate_answer("Q", "Context", "Answer")

    assert result["status"] == "FAIL"
    assert result["supported"] is False
    assert result["reason"].startswith("Validation error:")
