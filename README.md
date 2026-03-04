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

<br/>

> **An intelligent medical assistant that retrieves real scientific evidence from PubMed and generates structured, evidence-based answers using state-of-the-art LLMs.**

⚠️ _For educational purposes only. Not a substitute for professional medical advice._

</div>

---

## 🎯 Try the Live Demo

<div align="center">

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl.svg)](https://huggingface.co/spaces/zakaria-ch49/Medical-Question-Answering-with-Retrieval-Augmented-Generation_RAG)

</div>

---

## ✨ Features

- 🔍 **Real-time PubMed Search** — Fetches up-to-date scientific articles from NCBI's PubMed database
- 🧠 **Semantic Search with BGE Embeddings** — Uses `BAAI/bge-base-en`, a high-performance biomedical embedding model
- ⚡ **FAISS Vector Store** — Lightning-fast similarity search across thousands of medical documents
- 🤖 **LLM-powered Answers** — Generates structured, cited, evidence-based answers via OpenRouter
- 🌊 **Streaming Responses** — Real-time token-by-token streaming for a smooth user experience
- 🌍 **Multilingual** — Responds in the same language as the user's question
- 🐳 **Dockerized** — Ready for deployment anywhere

---

## 📐 Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (src/streamlit_app.py)    │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  PubMed Downloader      │  ←  src/download_pubmed.py
          │  (NCBI E-utilities API) │
          └────────────┬────────────┘
                       │  articles (title + abstract)
          ┌────────────▼────────────┐
          │  BGE Embedding Engine   │  ←  src/bio_clinical_embeddings.py
          │  BAAI/bge-base-en       │
          │  FAISS Vector Store     │
          └────────────┬────────────┘
                       │  top-k relevant documents
          ┌────────────▼────────────┐
          │  OpenRouter LLM         │  ←  src/open_router.py
          │  (Qwen / any model)     │
          └────────────┬────────────┘
                       │
                  Medical Answer
```

---

## 🔬 How It Works

| Step | Description |
|------|-------------|
| 1️⃣ **Query Translation** | The user's question is used to build a precise PubMed search query |
| 2️⃣ **PubMed Retrieval** | Relevant articles (title + abstract) are fetched from NCBI E-utilities API |
| 3️⃣ **Vectorization** | Articles are embedded with `BAAI/bge-base-en` and indexed in FAISS |
| 4️⃣ **Semantic Search** | Top-k most relevant documents retrieved using cosine similarity |
| 5️⃣ **LLM Generation** | Retrieved documents sent to LLM to generate a structured, cited answer |

---

## 📁 Project Structure

```
medical-rag-assistant/
├── src/
│   ├── streamlit_app.py            # Streamlit web interface (main entry point)
│   ├── main.py                     # CLI interface
│   ├── download_pubmed.py          # PubMed article downloader (NCBI API)
│   ├── bio_clinical_embeddings.py  # BGE embeddings + FAISS vector store
│   └── open_router.py              # OpenRouter LLM client (streaming + sync)
├── data/
│   ├── raw/                        # Static medical reference data
│   └── cache/                      # Downloaded PubMed articles (auto-generated)
├── .env.example                    # Template for environment variables
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose for easy deployment
├── Makefile                        # Developer shortcuts
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+**
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

### 1. Clone & configure

```bash
git clone https://github.com/zakaria-ch49/Medical-Question-Answering-with-RAG.git
cd Medical-Question-Answering-with-RAG

cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
# Streamlit web UI
streamlit run src/streamlit_app.py
```

Open your browser at **http://localhost:7860**

---

## 🐳 Docker Deployment

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | Your [OpenRouter](https://openrouter.ai) API key | ✅ Yes |
| `HF_TOKEN` | Hugging Face token (for model access) | ⚡ Optional |

---

## 📦 Tech Stack

| Technology | Role |
|------------|------|
| `Streamlit` | Interactive web UI |
| `LangChain` | RAG pipeline orchestration |
| `BAAI/bge-base-en` | Biomedical text embeddings |
| `FAISS` | Vector similarity search |
| `OpenRouter` | LLM API gateway (Qwen, GPT-4, etc.) |
| `PubMed NCBI API` | Scientific medical literature |
| `Docker` | Containerized deployment |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by **Zakaria CHADADI** for medical AI research

[![🚀 Try the Demo](https://img.shields.io/badge/🚀%20Try%20the%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/zakaria-ch49/Medical-Question-Answering-with-Retrieval-Augmented-Generation_RAG)

</div>
| `sentence-transformers` | Embedding model loading |
| `requests` | PubMed & OpenRouter API calls |
| `python-dotenv` | Environment variable management |

---

## 📄 License

See [LICENSE](LICENSE).
