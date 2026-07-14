# AI Knowledge Assistant

## Overview

AI Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that answers user questions based only on a provided knowledge base.

The system processes different document types, creates embeddings, stores them in a vector database, retrieves the most relevant information, and generates accurate answers using Google's Gemini model.

This project was developed as part of the AI Track Team Assignment.

---

## Features

- Document Loader
- Metadata Extraction
- Document Chunking
- Embedding Generation
- Vector Database Storage
- Query Rewriting
- Semantic Retrieval
- Context Builder
- Prompt Engineering
- Gemini Integration
- Response Validation
- Conversation Memory
- Conversation Export (TXT & Markdown)
- AI Evaluation Framework

---

## Supported File Types

- PDF
- DOCX
- TXT
- Markdown
- CSV

---

## Project Structure

```
AI-Knowledge-Assistant/
│
├── data/
├── exports/
├── evaluation_results/
├── tests/
│
├── src/
│   ├── knowledge_base/
│   │   ├── loader.py
│   │   ├── metadata.py
│   │   ├── chunker.py
│   │   └── __init__.py
│   │
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── query_rewriter.py
│   ├── retriever.py
│   ├── context_builder.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── validator.py
│   ├── memory.py
│   ├── export.py
│   ├── evaluation.py
│   └── main.py
│
├── vector_db/
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```

---

## AI Pipeline

```
Documents
      │
      ▼
Document Loader
      │
      ▼
Metadata Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
      │
      ▼
Vector Database
      │
      ▼
User Question
      │
      ▼
Query Rewriting
      │
      ▼
Semantic Retrieval
      │
      ▼
Context Builder
      │
      ▼
Prompt Builder
      │
      ▼
Gemini
      │
      ▼
Response Validation
      │
      ▼
Conversation Memory
      │
      ▼
Final Answer
      │
      ▼
Conversation Export
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Go to the project directory:

```bash
cd AI-Knowledge-Assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add your Gemini API key:

```
GEMINI_API_KEY=YOUR_API_KEY
```

---

## Running the Project

```bash
python src/main.py
```

---

## Export Feature

The assistant supports exporting conversations in:

- TXT
- Markdown (.md)

Exported files are automatically saved inside:

```
exports/
```

---

## Evaluation

The evaluation module measures:

- Retrieval Accuracy
- Grounded Responses
- Expected vs Actual Answers
- Overall Performance Score

---

## Technologies Used

- Python
- Google Gemini API
- ChromaDB
- Sentence Transformers
- LangChain (if used)
- Markdown
- Git
- GitHub

---

## Team Responsibilities

| Member | Responsibility |
|---------|---------------|
| Member 1 | Loader, Metadata, Chunking, Embeddings |
| Member 2 | Vector Database, Retrieval, Query Rewriting, Context Builder |
| Member 3 | Prompt Builder, Gemini Integration, Validation, Memory |
| Member 4 | Evaluation, Export, Documentation, Testing, Final Integration |

---

## Future Improvements

- Hybrid Search
- OCR Support
- Streaming Responses
- Multi-Query Retrieval
- Automatic Evaluation Reports
- Confidence Scoring

---

## License

This project was developed for educational purposes as part of the AI Track Team Assignment.