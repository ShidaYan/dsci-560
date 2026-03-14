# DSCI-560 Lab 9 — Custom Q&A Chatbot

**Course:** DSCI-560 Data Science Practicum | Spring 2026  
**Instructor:** Young H. Cho, Ph.D.

---

## Project Overview

A Question-and-Answer chatbot that reads PDF documents, converts them into vector embeddings, stores them in a FAISS vector database, and answers user questions using an LLM. Built in three versions:

- **Part 2** — OpenAI GPT + OpenAI Embeddings (terminal)
- **Part 3** — Local HuggingFace model + Local LLM, fully offline (terminal)
- **Part 4** — Flask web interface with PDF upload and chat window

---

## File Structure

```
lab9_chatbot/
├── app.py                        # Part 2 driver — terminal Q&A loop
├── pdf_extract.py                # PDF text extraction → SQLite
├── chunker.py                    # CharacterTextSplitter, size=500
├── vector_store.py               # OpenAI Embedding + FAISS index
├── conv_chain.py                 # ConversationBufferMemory chain
├── open_source/
│   ├── hf_embedder.py            # HuggingFace local embeddings
│   ├── local_llm.py              # ctransformers GGUF local LLM
│   ├── vector_store_os.py        # Offline FAISS index
│   ├── conv_chain_os.py          # Offline conversation chain
│   └── app_os.py                 # Part 3 driver — offline Q&A loop
├── web/
│   ├── app_web.py                # Flask backend (/upload + /chat)
│   ├── index.html                # Chat frontend
│   └── style.css                 # UI styles
├── pdfs/                         # Place PDF files here
├── data/                         # SQLite DB + FAISS index (auto-created)
├── models/                       # GGUF model files for Part 3
├── requirements.txt
└── .env                          # OPENAI_API_KEY (never commit)
```

---

## System Requirements

- Python 3.10 or above
- pip 23.0 or above
- RAM: 8 GB minimum (16 GB recommended for Part 3)
- Disk: 10 GB free (models can be 4–8 GB)

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd lab9_chatbot
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you encounter proxy-related errors, run:
> ```bash
> pip install "httpx[socks]" socksio
> ```

### 4. Configure API Key

Create a `.env` file in the project root:

```bash
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

### 5. Create required directories

```bash
mkdir -p pdfs data models
```

### 6. Add PDF files

```bash
cp /path/to/your/document.pdf pdfs/
```

---

## Running the Project

### Part 2 — OpenAI Terminal Mode

```bash
# First run: extract PDFs, build index, start chatbot
python app.py

# Subsequent runs: reload saved index (faster)
python app.py --reload
```

Type your question and press Enter. Type `exit` to quit.

### Part 3 — Offline Mode (Open-Source)

**Step 1:** Download a local GGUF model (~4 GB, one-time):

```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --local-dir models/
```

**Step 2:** Run the offline chatbot:

```bash
python open_source/app_os.py

# Or reload existing offline index:
python open_source/app_os.py --reload
```

No API key or internet connection required after model download.

### Part 4 — Web Interface

```bash
python web/app_web.py
```

Open browser and navigate to: `http://localhost:5000`

1. Click **Browse files** or drag-and-drop a PDF into the sidebar
2. Click **Analyse PDFs** — the server will extract, chunk, and embed the document
3. Type questions in the chat window and press Enter

---

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk size | 500 chars | CharacterTextSplitter chunk size |
| Chunk overlap | 50 chars | Overlap between consecutive chunks |
| Top-K retrieval | 4 | Number of chunks retrieved per query |
| LLM temperature | 0.0 | Deterministic output for Q&A |
| OpenAI embedding | text-embedding-ada-002 | Part 2 embedding model |
| HF embedding | all-MiniLM-L6-v2 | Part 3 embedding model (384-dim) |

---

## Troubleshooting

**`ValueError: document closed`**  
Fixed in `pdf_extract.py` — `page_count` is now saved before `doc.close()`.

**`unexpected keyword argument 'proxies'`**  
Version conflict between openai and langchain-openai. Run:
```bash
pip install openai==1.58.1 langchain-openai==0.2.14
```

**`ImportError: Using SOCKS proxy`**  
```bash
pip install "httpx[socks]" socksio
```

**`ImportError: cannot import name 'is_running_from_reloader'`**  
Caused by naming a file `html.py` which conflicts with Python's built-in `html` module. The web backend has been renamed to `app_web.py`.

**`Address already in use` (port 5000)**  
```bash
kill $(lsof -t -i:5000)
python web/app_web.py
```

**`openai.RateLimitError: 429 insufficient_quota`**  
OpenAI API quota exceeded. Add billing at: https://platform.openai.com/settings/billing  
Alternatively, use the offline Part 3 version: `python open_source/app_os.py`

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| openai | 1.58.1 | OpenAI API client |
| langchain | 0.3.14 | LLM orchestration framework |
| langchain-openai | 0.2.14 | LangChain OpenAI integration |
| langchain-community | 0.3.14 | FAISS, memory, loaders |
| faiss-cpu | 1.8.0 | Vector similarity search |
| pymupdf | 1.24.3 | PDF text extraction |
| tiktoken | 0.7.0 | Token counter |
| python-dotenv | 1.0.1 | Load .env variables |
| flask | 2.3.3 | Web backend |
| werkzeug | 2.3.7 | Flask dependency |
| sentence-transformers | 3.0.0 | HuggingFace local embeddings |
| transformers | 4.41.2 | HuggingFace model loading |
| torch | 2.3.0 | PyTorch inference backend |
| ctransformers | 0.2.27 | GGUF quantised LLM runner |
| huggingface-hub | 0.23.0 | Download models from HF Hub |

---

## Submission Checklist

- [ ] All code files (complete and runnable)
- [ ] `README.md` (this file)
- [ ] `meeting_notes_L9_<team_name>.pdf`
- [ ] GitHub commit history with each member's contributions
- [ ] Demo video showing full functionality
