from types import SimpleNamespace

from src import generator


def test_generate_text_returns_model_response(monkeypatch) -> None:
    def fake_generate_content(*, model: str, contents: str):
        assert model == "gemini-2.5-flash"
        assert contents == "Test prompt"
        return SimpleNamespace(text="Generated answer")

    monkeypatch.setattr(
        generator.client.models,
        "generate_content",
        fake_generate_content,
    )

    assert generator.generate_text("Test prompt") == "Generated answer"


def test_generate_text_handles_api_error(monkeypatch) -> None:
    def raise_error(**kwargs):
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(
        generator.client.models,
        "generate_content",
        raise_error,
    )

    result = generator.generate_text("Test prompt")

    assert result.startswith("Error generating text:")
    assert "API unavailable" in result
