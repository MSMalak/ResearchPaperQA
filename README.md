![CI](https://github.com/MSMalak/ResearchPaperQA/actions/workflows/tests.yml/badge.svg)

# 📄 ResearchPaperQA

> RAG pipeline for querying research papers from the terminal.

**Ask questions to research papers using a local RAG pipeline (PDF → FAISS → LLM).**

ResearchPaperQA is a lightweight Retrieval-Augmented Generation (RAG) demo that allows you to query research papers (PDFs) directly from the terminal using semantic search and language models.

---

## 🚀 Features

* 📚 PDF ingestion and chunking
* 🔍 Semantic retrieval with Sentence-Transformers + FAISS
* 🤖 Answer generation with:

  * Local HuggingFace models (default)
  * OpenAI models (optional)
* 🧩 Modular architecture (loader / embedder / vector store / generator)
* 🖥️ Clean CLI interface (`researchpaperqa`)
* 🔁 Reproducible indexing with cache control

---

## 📦 Installation

```bash
git clone https://github.com/MSMalak/ResearchPaperQA.git
cd ResearchPaperQA

python -m venv venv
source venv/bin/activate
pip install -e .
```

> The project is installed in *editable mode* so changes are picked up automatically.

---

## ▶️ Quickstart

Build the vector index and start querying papers:

```bash
researchpaperqa --documents data/sample_papers --recreate
```

Then ask questions directly in the terminal, for example:

* *“What is the main contribution of this paper?”*
* *“Which methodology is used?”*
* *“What problem does the paper address?”*

---

## 🖥️ UI Demo (optional)

An optional Streamlit interface is provided for quick interactive demos.

```bash
pip install -r requirements-ui.txt
streamlit run app.py
```

> Note: Local generation can be slower depending on hardware.
For faster responses, prefer the CLI with the OpenAI backend (`--generator openai`) if an API key is available.
---

## 🧠 How it works

1. Load PDFs from a directory
2. Split documents into semantic chunks
3. Embed chunks using Sentence-Transformers
4. Index embeddings with FAISS
5. Retrieve top-k relevant chunks for a query
6. Generate an answer conditioned on retrieved context

```
PDFs → Chunking → Embeddings → FAISS → Retrieval → Answer
```

---

## ⚙️ CLI Usage

```bash
researchpaperqa --help
```

```text
usage: researchpaperqa [-h] [--documents DOCUMENTS]
                       [--generator {local,openai}] [--recreate]

ResearchPaperQA — RAG chatbot over research PDFs
```

### Options

* `--documents` : path to a directory containing PDF files
* `--generator` : `local` (HuggingFace) or `openai`
* `--recreate`  : force rebuilding the vector index

---

## 🔐 Notes on security & reproducibility

* The FAISS index uses pickle-based serialization internally.
* Deserialization is enabled only for locally created indexes.
* Vector indexes and metadata are intentionally excluded from version control.

---

## 🧪 Project structure

```text
rag_chatbot/
├── loader.py        # PDF loading & chunking
├── embedder.py      # Embedding models
├── vectorstore.py   # FAISS index management
├── generator.py    # LLM backends
├── main.py          # CLI entry point
```

---

## 📌 Limitations & future work

* UI provided as an optional Streamlit demo (CLI remains the primary interface)
* Single-document indexing (for now)
* Potential extensions:

  * Multi-document comparison
  * Streaming answers
  * Web or notebook interface
  * Evaluation of retrieval quality

---

## 📄 License

MIT License



