# AgniAI

Standalone offline-first CLI chatbot for Agniveer-related questions using Ollama, local FAISS retrieval, and sentence-transformer embeddings.

## Setup

1. Install Ollama.
2. Pull and run a model:

```bash
ollama pull llama3
ollama run llama3
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Commands

```text
/ingest pdf <path>
/ingest url <url>
/ingest text <text>
/exit
```

## Example

```text
You: What is age limit for Agniveer?
AgniAI: 
Agniveer Age Eligibility:
- Minimum: 17.5 years
- Maximum: 21 years
(Source: Official Notification)
```

