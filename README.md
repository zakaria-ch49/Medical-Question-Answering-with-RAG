---
title: Medical Question Answering with Retrieval Augmented Generation RAG
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

<div align="center">

# 🏥 Medical RAG Assistant

### AI-Powered Medical Question Answering with Retrieval-Augmented Generation

[![🚀 Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/zakaria-ch49/Medical-Question-Answering-with-Retrieval-Augmented-Generation_RAG)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/zakaria-ch49/Medical-Question-Answering-with-RAG)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![CI Tests](https://github.com/zakaria-ch49/Medical-Question-Answering-with-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/zakaria-ch49/Medical-Question-Answering-with-RAG/actions/workflows/ci.yml)

</div>

---

## 🆕 What's New

| Update | Description |
|--------|-------------|
| 🔴 **openFDA dual-source** | Drug labels from the FDA API are now fetched **in parallel** with PubMed. If PubMed times out, FDA results are used automatically — and vice-versa. |
| 🔴 **Intelligent query cleaning** | FDA queries are automatically converted from PubMed MeSH syntax to plain FDA-compatible search terms, with progressive fallback variants. |
| 🔴 **PubMed resilience** | Retry on `ReadTimeout` / `ConnectTimeout` with exponential backoff. Failing batches are recursively split in two (up to 2 levels) instead of being dropped entirely. |
| 🔴 **Faster responses** | Batch size reduced from 100 → 20, timeout scaled per article (2 s/article), backoff reduced from 1.5 s → 0.5 s. Typical response time: **45–90 s** vs. previously 10 min. |
| 🔴 **Structured AI answers** | System prompt redesigned: responses now follow a strict **Answer / Key Points / Disclaimer** format — concise, professional, no emojis, full medical terminology, source citations. |
| 🔴 **Source badges in UI** | Documents from PubMed show a clickable PMID link; FDA drug labels show an orange **FDA Drug Label** badge. Metrics panel shows PubMed and FDA counts separately. |

<br/>

> **An intelligent medical assistant that retrieves real scientific evidence from PubMed and generates structured, evidence-based answers using state-of-the-art LLMs.**

⚠️ _For educational purposes only — not a substitute for professional medical advice._

</div>

---

## 🎯 Try the Live Demo

<div align="center">

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl.svg)](https://huggingface.co/spaces/zakaria-ch49/Medical-Question-Answering-with-Retrieval-Augmented-Generation_RAG)

</div>

---

## ✨ Features

| | Feature | Detail |
|---|---|---|
| 🔍 | **Real-time PubMed Search** | Fetches up-to-date scientific articles from NCBI’s PubMed database (E-utilities API) |
| 📊 | **openFDA Drug Labels** | Fetches official FDA drug label data in parallel with PubMed — indications, mechanism, dosage, warnings |
| 🧠 | **BGE Semantic Embeddings** | `BAAI/bge-base-en` — state-of-the-art biomedical embedding model |
| ⚡ | **FAISS Vector Store** | Lightning-fast similarity search across thousands of documents |
| 🌐 | **Auto Query Translation** | Converts any language / typo / vague question into a precise PubMed MeSH query |
| 🔄 | **Resilient Network Layer** | `HTTPAdapter` + `Retry` with exponential backoff, `ReadTimeout` retries, recursive batch splitting |
| 🤖 | **Structured AI Answers** | Fixed **Answer / Key Points / Disclaimer** format — concise, cited, professional medical terminology |
| 🌊 | **Streaming Responses** | Real-time token-by-token output for a smooth user experience |
| 🌍 | **Multilingual** | Responds in the same language as the user’s question |
| 🐳 | **Dockerized** | One-command deployment anywhere |

---

## 📐 Architecture

```
User Question (any language)
        │
        ▼
┌───────────────────────────────────────────────┐
│          Streamlit Web UI                     │
│          src/streamlit_app.py                 │
└──────────────────┬────────────────────────────┘
                   │
     ┌─────────────▼──────────────┐
     │   Query Translator         │   OpenRouter API
     │   (LLM → MeSH English)     │ ──────────────────
     └─────────────┬──────────────┘
                   │  optimised PubMed query
     ┌─────────────▼──────────────┐
     │   PubMed Downloader        │   src/download_pubmed.py
     │   NCBI E-utilities API     │   (batch fetch, up to 200 articles)
     └─────────────┬──────────────┘
                   │  title + abstract (preprocessed)
     ┌─────────────▼──────────────┐
     │   BGE Embedding Engine     │   src/bio_clinical_embeddings.py
     │   BAAI/bge-base-en         │
     │   FAISS Vector Index       │
     └─────────────┬──────────────┘
                   │  top-k relevant documents (BGE score ≤ 0.30)
     ┌─────────────▼──────────────┐
     │   OpenRouter LLM           │   src/open_router.py
     │   Qwen3-235B-A22B-Thinking │   (streaming SSE)
     └─────────────┬──────────────┘
                   │
         Structured Medical Answer
         (cited, evidence-graded, multilingual)
```

---

## 🔬 RAG Pipeline — Step by Step

| Step | Module | Description |
|------|--------|-----------|
| **1 — Query Translation** | `open_router.py` | User’s question (any language / typos) → precise PubMed MeSH query in English via LLM |
| **2 — Dual-Source Retrieval** | `download_pubmed.py` | **PubMed** (articles) + **openFDA** (drug labels) fetched in parallel — auto-fallback if one source fails |
| **3 — Preprocessing** | `download_pubmed.py` | Clean text: strip MeSH tags, normalize whitespace, remove special chars |
| **4 — Vectorization** | `bio_clinical_embeddings.py` | Embed all documents with `BAAI/bge-base-en` → FAISS index |
| **5 — Semantic Search** | `bio_clinical_embeddings.py` | Retrieve top-k documents by cosine similarity (BGE score threshold ≤ 0.30) |
| **6 — Answer Generation** | `open_router.py` | Stream structured **Answer / Key Points / Disclaimer** response via Qwen3-235B |

---

## 📁 Project Structure

```
medical-rag-assistant/
│
├── src/
│   ├── streamlit_app.py            # 🖥️  Streamlit web interface (main entry point)
│   ├── main.py                     # ⌨️  CLI interface (interactive terminal mode)
│   ├── download_pubmed.py          # 📥  PubMed batch downloader + text preprocessor
│   ├── bio_clinical_embeddings.py  # 🧠  BGE embeddings + FAISS vector store
│   └── open_router.py              # 🤖  OpenRouter LLM client (streaming + sync)
│
├── tests/
│   ├── conftest.py                 # Pytest configuration & shared fixtures
│   ├── test_download_pubmed.py     # Unit tests — PubMed downloader
│   ├── test_bio_clinical_embeddings.py  # Unit tests — embeddings & search
│   └── test_open_router.py         # Unit tests — OpenRouter client
│
├── data/
│   └── cache/                      # 💾  Auto-generated PubMed cache (git-ignored)
│
├── .github/workflows/ci.yml        # ✅  GitHub Actions CI (pytest on every push)
├── .env.example                    # 🔑  Environment variable template
├── Dockerfile                      # 🐳  Container image definition
├── docker-compose.yml              # 🐳  One-command local deployment
├── Makefile                        # 🛠️  Developer shortcuts (install, run, clean…)
├── requirements.txt                # 📦  Full Python dependencies
└── requirements-ci.txt             # 📦  Lightweight CI-only dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+**
- An [OpenRouter](https://openrouter.ai) API key — free tier available

### 1 · Clone & configure

```bash
git clone https://github.com/zakaria-ch49/Medical-Question-Answering-with-RAG.git
cd Medical-Question-Answering-with-RAG

cp .env.example .env
# Open .env and paste your OPENROUTER_API_KEY
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
# or: make install
```

### 3 · Launch the web UI

```bash
streamlit run src/streamlit_app.py
# or: make run
```

Open **http://localhost:8501** in your browser.

### 4 · (Optional) CLI mode

```bash
python src/main.py
# or: make cli
```

---

## 🐳 Docker Deployment

```bash
# Build and start (detached)
docker compose up --build -d

# Follow logs
docker compose logs -f

# Stop
docker compose down
```

The app is exposed on port **7860** (HuggingFace Spaces compatible).

---

## ✅ Running Tests

```bash
pip install -r requirements-ci.txt
pytest tests/ -v
```

The test suite uses mocks — no internet connection or GPU required.

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your [OpenRouter](https://openrouter.ai) API key | ✅ Yes |
| `HF_TOKEN` | Hugging Face token (optional, for private models) | ⚡ Optional |

Create a `.env` file from the template:

```bash
cp .env.example .env
```

---

## 📦 Tech Stack

| Technology | Version | Role |
|------------|---------|------|
| [Streamlit](https://streamlit.io) | latest | Interactive web UI |
| [LangChain](https://langchain.com) | latest | RAG pipeline orchestration |
| [BAAI/bge-base-en](https://huggingface.co/BAAI/bge-base-en) | — | Biomedical text embeddings |
| [FAISS](https://github.com/facebookresearch/faiss) | cpu | Vector similarity search |
| [OpenRouter](https://openrouter.ai) | — | Unified LLM API gateway |
| [Qwen3-235B-A22B-Thinking](https://openrouter.ai/qwen/qwen3-vl-235b-a22b-thinking) | 235B / 22B active | Answer generation (MoE, streaming) |
| [PubMed NCBI API](https://www.ncbi.nlm.nih.gov/home/develop/api/) | E-utilities | Scientific medical literature |
| [openFDA API](https://open.fda.gov/apis/drug/label/) | drug/label | Official FDA drug label database |
| [Docker](https://docker.com) | — | Containerized deployment |
| [pytest](https://pytest.org) | — | Unit testing |

> **LLM — `qwen/qwen3-vl-235b-a22b-thinking`**
> - **235 billion** total parameters · **22 billion** active (Mixture of Experts)
> - Advanced chain-of-thought reasoning · Multilingual · Strong medical comprehension

---

## ⚠️ Limitations

- 🌐 **Internet required** — PubMed and openFDA articles are fetched live; no offline mode
- 🐢 **Cold start** — BGE embedding model loads on first query (~30 s on CPU)
- 📄 **Abstracts only** — Full-text articles are not retrieved, only titles & abstracts
- 🔑 **API key required** — An OpenRouter key is needed to generate answers
- 🏥 **Not for clinical use** — For educational and research purposes only; always consult a healthcare professional
- 🌍 **English-biased** — PubMed is predominantly in English; non-English queries may return fewer results

---

## 🗺️ Roadmap

| Status | Feature |
|--------|---------|
| ✅ Done | PubMed real-time retrieval |
| ✅ Done | openFDA drug label retrieval (parallel, auto-fallback) |
| ✅ Done | BGE semantic search with FAISS |
| ✅ Done | Streaming LLM answers |
| ✅ Done | Structured AI responses (Answer / Key Points / Disclaimer) |
| ✅ Done | PubMed network resilience (retry + backoff + batch splitting) |
| ✅ Done | Source badges in UI (PubMed vs FDA) |
| ✅ Done | Docker deployment on Hugging Face Spaces |
| ✅ Done | CI/CD with GitHub Actions |
| 🔄 Planned | Full-text article retrieval (PMC Open Access) |
| 🔄 Planned | Chat history / conversation memory |
| 🔄 Planned | Export answers to PDF |
| 🔄 Planned | GPU acceleration for embeddings |

> 💡 **Contributions welcome!** Feel free to open issues or pull requests.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Zakaria CHADADI** for medical AI research

[![🚀 Try the Demo](https://img.shields.io/badge/🚀%20Try%20the%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/zakaria-ch49/Medical-Question-Answering-with-Retrieval-Augmented-Generation_RAG)

</div>
