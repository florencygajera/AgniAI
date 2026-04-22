# AgniAI — Offline Agniveer Chatbot

A fully local, offline-first CLI chatbot that answers **Agniveer / Agnipath recruitment** questions using:

- 🦙 **Ollama** — runs LLMs (Llama 3, Mistral, Phi-3, …) 100% locally
- 🔍 **FAISS** — fast vector similarity search
- 🧠 **Sentence Transformers** — local embeddings (no API needed)
- 📄 **Dynamic RAG** — ingest PDFs, URLs, or text at any time


## Requirements

| Tool | Minimum Version |
|------|----------------|
| Python | 3.9+ |
| Ollama | 0.1.x+ |
| RAM | 8 GB (16 GB recommended) |
| Disk | ~5 GB for model weights |


## Step-by-step Setup

### 1 — Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from https://ollama.com/download

### 2 — Pull a local LLM

Choose one (Llama 3 is recommended):
```bash
ollama pull llama3       # ~4.7 GB — best quality
ollama pull mistral      # ~4.1 GB — fast & capable
ollama pull phi3         # ~2.3 GB — lightest option
```

### 3 — Start Ollama

```bash
ollama serve
```

Keep this terminal open. AgniAI calls it on `http://localhost:11434`.

### 4 — Clone / unzip AgniAI

```bash
cd agniai/
```

### 5 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 6 — Install Python dependencies

```bash
pip install -r requirements.txt
```

The first run will automatically download the embedding model (~90 MB).

### 7 — Run AgniAI

```bash
python main.py
```


## Ingesting Knowledge

AgniAI starts empty. Feed it documents before asking questions.

### Ingest a PDF (offline notification PDFs, circulars)
```
/ingest pdf /path/to/agniveer_notification.pdf
```

### Ingest a website
```
/ingest url https://joinindianarmy.nic.in/
```

### Ingest a text file
```
/ingest txt /path/to/agniveer_details.txt
```

### Ingest raw text
```
/ingest text Agniveer minimum age is 17.5 years and maximum is 21 years.
```


## Chat Commands

| Command | Action |
|---------|--------|
| `/ingest pdf <path>` | Ingest a PDF |
| `/ingest url <url>` | Ingest a webpage |
| `/ingest txt <path>` | Ingest a .txt file |
| `/ingest text <content>` | Ingest raw text |
| `/sources` | List all ingested sources |
| `/stats` | Show index vector count |
| `/clear` | Clear conversation memory |
| `/reset` | ⚠ Delete entire knowledge base |
| `/model <name>` | Switch Ollama model mid-session |
| `/help` | Show help |
| `/exit` or `/quit` | Exit |


## Example Session

```
You: What is the age limit for Agniveer?
AgniAI:
  Agniveer Age Eligibility:
  • Minimum age: 17.5 years
  • Maximum age: 21 years
  (Relaxation may apply for specific categories as per official notification.)
  📌 Source: /home/user/agniveer_notification.pdf

You: What is the salary?
AgniAI:
  Agniveer Monthly Package:
  • Year 1: ₹30,000/month
  • Year 2: ₹33,000/month
  • Year 3: ₹36,500/month
  • Year 4: ₹40,000/month
  • Seva Nidhi corpus (after 4 years): ~₹11.71 lakh
  📌 Source: https://joinindianarmy.nic.in/
```


## Project Structure

```
agniai/
├── main.py          # CLI chat loop + command dispatcher
├── rag.py           # Embeddings, FAISS search, Ollama LLM calls
├── ingest.py        # PDF / URL / text ingestion pipeline
├── memory.py        # Sliding-window conversation history
├── config.py        # All configuration constants
├── requirements.txt
├── data/            # (auto-created) raw data store
└── index/
    ├── agni.index   # FAISS binary index (auto-created)
    └── docstore.json# Chunk metadata store (auto-created)
```


## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull llama3` (or whichever model) |
| Slow first response | Normal — model loads into RAM on first call |
| "No text extracted" from PDF | PDF is image-based; use OCR tools first |
| Empty answers | Ingest relevant documents first with `/ingest` |


## Privacy

All computation happens **on your machine**. No data is sent to any cloud service.