# VidhiSetu (विधिसेतु) — Legal AI RAG & Retrieval Engine

**VidhiSetu** is an authoritative, source-grounded Legal AI Retrieval-Augmented Generation (RAG) system engineered for Indian legislation and statutory provisions. It indexes and searches primary legislative sources (including Bharatiya Nyaya Sanhita 2023, Bharatiya Nagarik Suraksha Sanhita, Bharatiya Sakshya Adhiniyam, Indian Penal Code, Code of Civil Procedure 1908, Indian Contract Act 1872, Arms Act 1959, and other major Acts) using hybrid lexical and dense multilingual semantic retrieval.

---

## 🏛️ System Architecture

1. **Ingestion & Section Parsing**:
   - Native PyMuPDF text extraction across 33 central Acts and the IPC sections dataset.
   - 3,717 parsed statutory sections preserving chapter hierarchies, section titles, and page provenance.
   - 5,995 legal-aware chunks segmented along statutory boundaries (subsections, clauses, provisos, explanations).

2. **Hybrid Retrieval (RRF)**:
   - **BM25 Lexical Engine**: Exact statutory terms, section numbers, and legal nomenclature.
   - **BGE-M3 Multilingual Semantic Engine**: 1,024-dimensional dense vector embeddings (`BAAI/bge-m3`) with cosine similarity.
   - **Reciprocal Rank Fusion (RRF)**: Combines ranking lists ($k = 60$) with exact section metadata candidate matching and deduplication.

3. **Source-Grounded QA Engine**:
   - Zero-hallucination policy: Legal statements are tied strictly to retrieved legislative text.
   - Insufficient evidence detection: Explicitly flags out-of-scope or absent legal queries without fabricating citations.
   - Path sanitization: Prevents exposure of host filesystem paths or credentials.

4. **FastAPI Backend (Phase 6)**:
   - Lifespan singleton initialization: In-memory embeddings and BM25 index load once at startup.
   - Role-specific response policies:
     - **Civilian (`/api/v1/civilian/ask`)**: Plain language explanations, actionable takeaways, and statutory notices.
     - **Student (`/api/v1/student/ask`)**: Statutory analysis, legislative context, and pedagogical principles.
     - **Lawyer (`/api/v1/lawyer/research`)**: Verbatim statutory excerpts, citation metadata, and Act-level filtering.

---

## 🚀 Quickstart & Setup (Windows PowerShell)

### 1. Prerequisites
- Windows 10/11
- Python 3.13.x or 3.11+
- PowerShell

### 2. Activate Virtual Environment & Install Dependencies
Open PowerShell in the project directory:

```powershell
# Activate the existing virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade pip and install required dependencies
.\venv\Scripts\pip.exe install -r requirements.txt
```

### 3. Environment Configuration
Copy the sample environment file:

```powershell
Copy-Item .env.example .env
```

---

## 🌐 Starting the FastAPI Application

Start the development server using Uvicorn:

```powershell
# Start Uvicorn on localhost:8000
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Once running:
- **Interactive OpenAPI Docs (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Docs (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🧪 Running Tests

### 1. Run FastAPI Integration Tests
```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

### 2. Run Retrieval Engine Unit Tests
```powershell
.\venv\Scripts\python.exe -m unittest discover -s retrieval -p "test_*.py"
```

### 3. Run Empirical Retrieval Evaluation Suite
```powershell
.\venv\Scripts\python.exe evaluation/evaluate_retrieval.py
```

### 4. Run Live API Verification Script
```powershell
.\venv\Scripts\python.exe scripts/verify_api_live.py
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Sample Query |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health & model status | N/A |
| `POST` | `/api/v1/search` | Raw hybrid search (BM25 + BGE-M3) | `{"query": "breach of contract", "top_k": 5}` |
| `POST` | `/api/v1/civilian/ask` | Plain language civilian explanation | `{"query": "Section 73 of Indian Contract Act", "top_k": 3}` |
| `POST` | `/api/v1/student/ask` | Academic & statutory analysis | `{"query": "Section 3 Arms Act 1959", "top_k": 3}` |
| `POST` | `/api/v1/lawyer/research` | Comprehensive lawyer research dossier | `{"query": "culpable homicide", "top_k": 5, "act_filter": "BNS"}` |

---

## ⚠️ Limitations & Disclaimers

> [!IMPORTANT]
> **Legal Notice**: VidhiSetu is an educational and technical research retrieval system. Outputs are synthesized directly from local indexed legislative documents and do not constitute formal legal counsel or advocate-client representation. Practicing advocates and litigants must independently verify provisions against official gazette notifications.
