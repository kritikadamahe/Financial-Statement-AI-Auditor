# FinAuditAI — AI-Powered Financial Statement Auditor

<div align="center">

**Audit Smarter, Not Harder.**

An enterprise-grade AI audit copilot that analyzes financial statements, detects anomalies, verifies GAAP/IFRS compliance, benchmarks against industry standards, and lets you chat with your financial data — all in one sleek dashboard.

</div>

---

## What is FinAuditAI?

FinAuditAI is an end-to-end AI-powered audit preparation platform. Upload your company's financial statements (Balance Sheet, Income Statement, Cash Flow — in `.csv` or `.xlsx`), and the system will instantly:

1. **Parse & Extract** structured financial metrics across multiple fiscal years.
2. **Calculate Key Ratios** — Current Ratio, Gross Margin, Debt-to-Equity, Net Profit Margin.
3. **Detect Anomalies** — Flag revenue spikes without corresponding cash flow, runaway expenses, and large year-over-year swings.
4. **Verify Compliance** — Run an AI-powered GAAP/IFRS structural check on the data.
5. **Benchmark Against Peers** — Compare the company's ratios against industry averages (Tech, Manufacturing, Retail, Healthcare).
6. **Generate Auditor Questions** — Produce 3–5 critical, management-facing audit questions tailored to the specific findings.
7. **Chat with the Data** — Ask natural language follow-up questions and get AI-backed answers grounded in the uploaded spreadsheet.
8. **Export a PDF Report** — One-click export of the entire dashboard (charts, anomalies, questions) to a professional PDF.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
│  │  Upload   │  │  Charts  │  │ Anomaly  │  │  AI Copilot   │   │
│  │  Page     │  │ AreaChart│  │  Grid    │  │  Chat Widget  │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘   │
│       │              │             │                │           │
│       └──────────────┴─────────────┴────────────────┘           │
│                          │  Axios HTTP                          │
└──────────────────────────┼──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │  /api/upload  │  │ /api/history   │  │ /api/chat/{id}       │  │
│  │  Multi-file   │  │ List / Detail  │  │ Natural Language Q&A │  │
│  └──────┬───────┘  └───────┬────────┘  └──────────┬───────────┘  │
│         │                  │                      │              │
│  ┌──────▼──────────────────▼──────────────────────▼───────────┐  │
│  │                   Service Layer                            │  │
│  │  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │  │
│  │  │ Parser  │  │ Analysis  │  │ AI (Groq)│  │Benchmarks │  │  │
│  │  │ csv/xlsx│  │ Ratios &  │  │ Llama3.3 │  │ Industry  │  │  │
│  │  │         │  │ Anomalies │  │ 70B      │  │ Averages  │  │  │
│  │  └─────────┘  └───────────┘  └──────────┘  └───────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                          │                                       │
│                   ┌──────▼──────┐                                │
│                   │   SQLite    │                                │
│                   │  Database   │                                │
│                   └─────────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Upload** → User selects one or more financial files + an industry from the frontend.
2. **Parse** → `parser.py` reads CSV/Excel into a Pandas DataFrame.
3. **Analyze** → `analysis.py` extracts metrics, calculates ratios, and detects anomalies.
4. **AI Layer** → `ai.py` sends findings to Groq (Llama 3.3 70B) for audit questions, compliance checks, and chat.
5. **Persist** → Results are stored in SQLite so users can revisit past audits.
6. **Render** → The frontend displays everything in a Groww-inspired dashboard with interactive charts, data grids, and a floating chat widget.

---

## Features

### Core Analysis
| Feature | Description |
|---|---|
| **Multi-Document Upload** | Upload multiple files (Balance Sheet + Income Statement + Cash Flow) simultaneously for cross-referencing |
| **Automated Ratio Mapping** | Calculates Current Ratio, Gross Margin, Debt-to-Equity, and Net Profit Margin across all fiscal years |
| **Anomaly Detection** | Rule-based engine flags revenue/cash flow divergence, expense spikes, and >40% year-over-year swings |

### AI-Powered
| Feature | Description |
|---|---|
| **Audit Question Generation** | Groq Llama 3.3 70B generates 3–5 critical, context-specific auditor questions |
| **GAAP/IFRS Compliance Check** | AI scans the financial structure for missing line items and formatting violations |
| **"Chat with Financials"** | Natural language copilot that answers follow-up questions grounded in your uploaded data |

### Industry Intelligence
| Feature | Description |
|---|---|
| **Industry Benchmarking** | Compare ratios against industry averages (Technology, Manufacturing, Retail, Healthcare) |
| **Reference Lines on Charts** | Dashed benchmark lines are overlaid directly on the AreaChart for instant visual comparison |

### Workflow & Export
| Feature | Description |
|---|---|
| **Interactive Tick-Marks** | Click anomalies and questions to cycle through Pending → Investigating → Cleared |
| **Audit History Sidebar** | All past analyses are saved and retrievable from a collapsible sidebar |
| **One-Click PDF Export** | Export the entire dashboard (charts, grids, questions) to a professional PDF via `html2pdf.js` |

### UI/UX
| Feature | Description |
|---|---|
| **Groww-Inspired Design** | Clean, card-based, minimalist fintech aesthetic with soft shadows and emerald accents |
| **Light & Dark Mode** | Professional theme toggle with optimized palettes for both modes |
| **Gradient AreaCharts** | Recharts AreaChart with smooth gradient fills under each trend line |
| **Animated Background** | Subtle moving DottedSurface background for visual depth |

---

## Tech Stack

### Frontend
| Technology | Purpose |
|---|---|
| **Next.js 16** (React 19, TypeScript) | App framework, routing, SSR |
| **Tailwind CSS v4** | Utility-first styling |
| **Recharts** | AreaChart, ReferenceLine, Tooltip, Legend |
| **Lucide React** | Icon library |
| **Axios** | HTTP client for API calls |
| **html2pdf.js** | Client-side PDF generation |
| **next-themes** | Light/Dark mode provider |

### Backend
| Technology | Purpose |
|---|---|
| **FastAPI** (Python 3.11) | REST API framework |
| **Groq SDK** | LLM inference (Llama 3.3 70B Versatile) |
| **SQLAlchemy** | ORM for SQLite persistence |
| **Pydantic** | Request/response validation |
| **Pandas + openpyxl** | CSV/Excel parsing and data manipulation |
| **python-dotenv** | Environment variable management |

### Infrastructure
| Technology | Purpose |
|---|---|
| **SQLite** | Lightweight, zero-config database |
| **Uvicorn** | ASGI server for FastAPI |

---

## Installation & Setup

### Prerequisites
- **Node.js** v18+ and **npm**
- **Python** 3.9+
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/kritikadamahe/Financial-Statement-AI-Auditor.git
cd Financial-Statement-AI-Auditor
```

### 2. Backend Setup (FastAPI + Groq)

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install groq

# Start the server
uvicorn main:app --port 8000
```

> **Note:** Create a `.env` file in the `backend/` directory and add your Groq API key:
> ```env
> GROQ_API_KEY=your_groq_api_key_here
> ```

The API will be available at **http://localhost:8000**. You can view the interactive docs at **http://localhost:8000/docs**.

### 3. Frontend Setup (Next.js)

Open a **new terminal** tab:

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The application will be available at **http://localhost:3000**.

### 4. Using the Application

1. Open **http://localhost:3000** in your browser.
2. Select an **Industry** from the dropdown (Technology, Manufacturing, Retail, or Healthcare).
3. Upload one or more financial statement files (`.csv`, `.xlsx`, `.xls`).
4. Click **"Run AI Audit"** and wait for the analysis to complete.
5. Explore the dashboard:
   - View the **AreaChart** with gradient fills and industry benchmark reference lines.
   - Review **Flagged Anomalies** in the data grid and toggle their status.
   - Read **AI Auditor Queries** in the right panel and check them off.
   - Check the **GAAP/IFRS Compliance** panel for any structural issues.
   - Click **"AI Copilot"** to chat with your financial data.
   - Click **"Export PDF"** to download a professional audit report.

---

## Evaluation & Research Pipeline

FinAuditAI includes a robust evaluation and research pipeline to generate synthetic training datasets, benchmark LLM parsing accuracy, and analyze model performance.

### 1. 1,200-Trial Monte Carlo ML Simulation
To resolve data scarcity, we simulate a realistic corporate population (1,000 normal + 200 fraudulent companies across 5 historical fraud types). To generate a new synthetic cohort:
```bash
# From the project root directory
python -m monte_carlo.main
```
This generates:
* `monte_carlo/synthetic_dataset.csv` (the raw panel dataset).
* `monte_carlo/dataset_report.md` (methodology and validation audit report).
* Visualization density curves and heatmap plots stored directly in `monte_carlo/`.

### 2. Population-Level Model Training & Evaluation
To train the anomaly models (Isolation Forest, Local Outlier Factor, One-Class SVM, and PyTorch LSTM Autoencoder) on the normal baseline population and evaluate their performance on the fraudulent cohort:
```bash
cd backend
python evaluate_ml.py
```
This outputs classification reports, confusion matrices, ROC curves, and per-layer breakdown charts at **300 DPI** inside `backend/paper_assets/`.

### 3. LLM Extraction Accuracy Benchmark
To test cell-level accuracy of the LLM parser against ground-truth spreadsheets across 4 sectors (Tech, Manufacturing, Retail, Healthcare):
```bash
cd backend
python evaluate_extraction.py
```
This saves the extraction accuracy bar chart to `backend/paper_assets/extraction_accuracy.png`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/upload` | Upload files (multipart) + industry (form field) → Full analysis |
| `GET` | `/api/history` | List all past audit records (lightweight) |
| `GET` | `/api/history/{id}` | Get full analysis payload for a specific record |
| `POST` | `/api/chat/{id}` | Send a natural language query about a specific audit record |

---

## Project Structure

```
Financial-Statement-AI-Auditor/
├── backend/
│   ├── main.py              # FastAPI app, routes, CORS
│   ├── models.py             # SQLAlchemy ORM models
│   ├── schemas.py            # Pydantic request/response schemas
│   ├── database.py           # SQLite engine & session
│   ├── requirements.txt      # Python dependencies
│   └── services/
│       ├── parser.py          # CSV/Excel file parsing
│       ├── analysis.py        # Ratio calculation, anomaly detection, benchmarks
│       └── ai.py              # Groq LLM integration (audit questions, compliance, chat)
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Main application UI (upload, dashboard, chat)
│   │   │   ├── layout.tsx     # Root layout with theme provider
│   │   │   └── globals.css    # Global styles and CSS variables
│   │   └── components/        # Reusable UI components (ThemeToggle, DottedSurface, etc.)
│   ├── package.json
│   └── tailwind.config.ts
│
└── README.md
```

---

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter bugs or want to suggest new features.

---

<div align="center">
  <strong>Built to empower finance teams and auditors with AI speed, precision, and insights.</strong>
  <br/>
  <sub>Powered by Groq • Llama 3.3 70B • Next.js • FastAPI</sub>
</div>
