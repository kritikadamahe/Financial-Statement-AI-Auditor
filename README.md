# FinAuditAI: Financial Statement AI Auditor

FinAuditAI is an AI-powered audit preparation agent designed to automate and enhance the financial auditing process. By analyzing financial statements (Excel/CSV), it instantly detects anomalies, performs detailed ratio analysis, and generates professional, auditor-style queries to accelerate audit readiness.

## ?? Features

- **Intelligent Anomaly Detection**: Uses sophisticated algorithms to flag inconsistencies, outliers, and potential discrepancies in uploaded financial records.
- **Automated Ratio Analysis**: Automatically calculates key financial ratios such as liquidity, solvency, and profitability.
- **Proactive Auditor Questions**: Generates a tailored list of questions that professional auditors are likely to ask based on the provided data.
- **Elegant & Interactive UI**: 
  - Responsive, modern dashboard built with **Next.js 15+** and **Tailwind CSS v4**.
  - Animated 3D background powered by **Three.js** (`DottedSurface`).
  - Eye-catching typography with **Framer Motion** (`GradientText`).
  - Seamless native **Dark Mode & Light Mode** support.
- **History Tracking**: Automatically saves previous analyses for quick retrieval, allowing users to track auditing history.

## ?? Tech Stack

### Frontend
- **Framework**: [Next.js](https://nextjs.org/) (React, TypeScript)
- **Styling**: [Tailwind CSS v4](https://tailwindcss.com/)
- **Components & Animations**: [Framer Motion](https://www.framer.com/motion/), [lucide-react](https://lucide.dev/), [Recharts](https://recharts.org/)
- **3D Graphics**: [Three.js](https://threejs.org/)
- **Data Fetching**: [Axios](https://axios-http.com/)

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Database**: SQLite (SQLAlchemy & Pydantic)
- **AI Processing**: Internal parsing and ML-based analysis services

## ?? Installation & Setup

Ensure you have **Node.js (v18+)** and **Python (v3.8+)** installed.

### 1. Clone the repository
```bash
git clone https://github.com/kritikadamahe/Financial-Statement-AI-Auditor.git
cd Financial-Statement-AI-Auditor
```

### 2. Set up the Backend (FastAPI)
Navigate to the backend folder and create a virtual environment:
```bash
cd backend
python -m venv venv
# On Windows: venv\Scripts\activate
# On Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
The FastAPI backend will start on `http://localhost:8000`.

### 3. Set up the Frontend (Next.js)
Open a new terminal tab and navigate to the frontend directory:
```bash
cd frontend
npm install
npm run dev
```
The Next.js application will be available at `http://localhost:3000`.

## ?? UI/UX Features
- **Global Theme Toggle**: Effortlessly switch between dark (pure black) and light (white) themes.
- **3D Background**: Subtly moving dotted grid responding statically behind the interface.
- **Animated Text Highlights**: Gradient borders and glowing text for key concepts using Motion.
- **Serif Typography**: Custom global `"Book Antiqua"` layout for a trusted, professional aesthetic.

## ?? Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter bugs or want to suggest new features.

---
*Built to empower finance teams and auditors with AI speed and insights.*
