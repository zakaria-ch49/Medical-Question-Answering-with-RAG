.PHONY: install run cli docker-up docker-down clean help

# ─── Variables ─────────────────────────────────────────────────────────────────
PYTHON      := python3
PIP         := pip3
SRC         := src
APP         := $(SRC)/streamlit_app.py
CLI         := $(SRC)/main.py
IMAGE_NAME  := medical-rag-assistant

# ─── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🏥  Medical RAG Assistant — Available Commands"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install     Install Python dependencies"
	@echo "  make run         Launch the Streamlit web UI"
	@echo "  make cli         Launch the CLI interface"
	@echo "  make docker-up   Build & start with Docker Compose"
	@echo "  make docker-down Stop Docker containers"
	@echo "  make clean       Remove caches and temporary files"
	@echo ""

# ─── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ─── Run ───────────────────────────────────────────────────────────────────────
run:
	streamlit run $(APP) --server.port=8501

cli:
	$(PYTHON) $(CLI)

# ─── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ─── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cache nettoyé"
