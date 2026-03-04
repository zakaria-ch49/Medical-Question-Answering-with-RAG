# 🏥 Medical RAG Assistant

> **Retrieval-Augmented Generation** system for medical question answering, powered by **PubMed**, **FAISS**, **BGE embeddings**, and **OpenRouter LLM**.

⚠️ _For educational purposes only. Not a substitute for professional medical advice._

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

## 📁 Project Structure

```
medical-rag-assistant/
├── src/                            # Core application code
│   ├── __init__.py
│   ├── streamlit_app.py            # Streamlit web interface (main entry point)
│   ├── main.py                     # CLI interface
│   ├── download_pubmed.py          # PubMed article downloader (NCBI API)
│   ├── bio_clinical_embeddings.py  # BGE embeddings + FAISS vector store
│   └── open_router.py              # OpenRouter LLM client (streaming + sync)
│
├── data/
│   ├── raw/                        # Static medical reference data
│   └── cache/                      # Downloaded PubMed articles (auto-generated)
│
├── docs/
│   └── report/                     # Project reports and documentation
│
├── tests/                          # Unit and integration tests
│   └── __init__.py
│
├── .env                            # API keys (not committed to git)
├── .env.example                    # Template for environment variables
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Docker Compose for easy deployment
├── Makefile                        # Developer shortcuts
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) API key

### 1. Clone & configure

```bash
git clone <repository-url>
cd medical-rag-assistant

cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Install dependencies

```bash
make install
# or manually:
pip install -r requirements.txt
```

### 3. Run the application

```bash
# Streamlit web UI
make run

# CLI interface
make cli
```

Open your browser at **http://localhost:8501**

---

## 🐳 Docker

```bash
# Build and start with Docker Compose
make docker-up

# Stop
make docker-down
```

Or manually:

```bash
docker build -t medical-rag-assistant .
docker run -p 8501:8501 --env-file .env medical-rag-assistant
```

---

## ⚙️ Configuration

| Variable | Description | Required |
|---|---|---|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | ✅ Yes |

---

## 🔬 How It Works

1. **Query Translation** — The user's question is translated into a precise PubMed search query using the LLM.
2. **PubMed Retrieval** — Relevant articles (title + abstract) are fetched from NCBI using the E-utilities API.
3. **Vectorization** — Articles are embedded using `BAAI/bge-base-en` (a high-quality biomedical English model) and indexed in FAISS.
4. **Semantic Search** — The top-k most relevant documents are retrieved using cosine similarity.
5. **LLM Generation** — The retrieved documents are passed to the LLM via OpenRouter to generate a structured, evidence-based answer.

---

## 📦 Key Dependencies

| Package | Role |
|---|---|
| `streamlit` | Web UI |
| `langchain` + `langchain-community` | RAG pipeline orchestration |
| `langchain-huggingface` | BGE embedding integration |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Embedding model loading |
| `requests` | PubMed & OpenRouter API calls |
| `python-dotenv` | Environment variable management |

---

## 📄 License

See [LICENSE](LICENSE).
