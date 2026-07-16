# Bonus Features

## Implemented

1. **Hybrid Search:** semantic ChromaDB retrieval and BM25 keyword retrieval are fused using Reciprocal Rank Fusion.
2. **Document Summarization During Indexing:** an extractive summary is created and indexed for every loaded document part.
3. **Multi-query Retrieval:** each question is expanded into complementary retrieval variants and results are fused.
4. **Confidence Scoring:** citation quality and validator support are combined into answer-level confidence.
5. **Streaming AI Responses:** enabled by default and toggleable using the `stream` command.
6. **Prompt Strategy Comparison:** `python -m src.prompt_comparison` exports CSV and Markdown comparisons.
7. **Automatic Evaluation Reports:** `python -m src.evaluation_runner` evaluates 25 questions and exports reports.
8. **OCR for Image-based PDFs:** pages without native text are rendered and processed with Tesseract OCR.
9. **Retrieval Decision Agent:** routes follow-up questions to conversation memory when document retrieval is unnecessary.

## Demonstration Commands

```bash
python -m src.main
python -m src.evaluation_runner
python -m src.prompt_comparison
python -m pytest -q
```
