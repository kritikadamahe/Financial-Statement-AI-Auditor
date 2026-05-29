"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import {
  UploadCloud, FileText, AlertTriangle, CheckCircle,
  BrainCircuit, Activity, BarChart3, History, X, Clock,
  ChevronRight, PlusCircle, TrendingUp, ShieldAlert,
  CheckSquare, Square, AlertCircle, Download, MessageSquare, 
  Send, ShieldCheck, FileCheck2
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine
} from "recharts";
import { ThemeToggle } from "@/components/ui/theme-toggle";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── Types ────────────────────────────────────────────────────────────────────

interface RatioResult {
  name: string;
  value: number;
  year: string;
}

interface AnomalyResult {
  description: string;
  severity: string;
  related_metrics: string[];
}

interface AnalysisResult {
  id?: number;
  filename?: string;
  upload_date?: string;
  ratios: RatioResult[];
  anomalies: AnomalyResult[];
  audit_questions: string[];
  raw_data_summary: { years_analyzed: string[] };
  industry_benchmarks?: Record<string, number>;
  compliance_flags?: string[];
}

interface HistoryItem {
  id: number;
  filename: string;
  upload_date: string;
  status: string;
  anomaly_count: number;
  ratio_count: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function timeAgo(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function severityColor(severity: string) {
  if (severity === "High") return "bg-rose-100 text-rose-700 dark:bg-rose-500/10 dark:text-rose-400 border-rose-200 dark:border-rose-500/20";
  if (severity === "Medium") return "bg-amber-100 text-amber-700 dark:bg-amber-500/10 dark:text-amber-400 border-amber-200 dark:border-amber-500/20";
  return "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400 border-emerald-200 dark:border-emerald-500/20";
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [industry, setIndustry] = useState("Technology");
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState<number | null>(null);

  // Chart metric filter
  const [activeMetrics, setActiveMetrics] = useState<Set<string>>(new Set(["Gross Margin (%)", "Current Ratio", "Debt-to-Equity Ratio", "Net Profit Margin (%)"]));

  const METRIC_CONFIG: Record<string, { color: string; gradient: string }> = {
    "Gross Margin (%)": { color: "#10b981", gradient: "url(#colorGross)" },
    "Current Ratio": { color: "#3b82f6", gradient: "url(#colorCurrent)" },
    "Debt-to-Equity Ratio": { color: "#f43f5e", gradient: "url(#colorDebt)" },
    "Net Profit Margin (%)": { color: "#a855f7", gradient: "url(#colorProfit)" },
  };

  const toggleMetric = (metric: string) => {
    setActiveMetrics(prev => {
      const next = new Set(prev);
      if (next.has(metric)) {
        if (next.size > 1) next.delete(metric);
      } else {
        next.add(metric);
      }
      return next;
    });
  };

  const showAllMetrics = () => setActiveMetrics(new Set(Object.keys(METRIC_CONFIG)));

  // Chat state
  const [chatOpen, setChatOpen] = useState(false);
  const [chatQuery, setChatQuery] = useState("");
  const [chatHistory, setChatHistory] = useState<{role: string, content: string}[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Tick-mark state
  const [statusMarks, setStatusMarks] = useState<Record<string, "pending" | "investigating" | "cleared">>({});

  const toggleStatus = (id: string) => {
    setStatusMarks(prev => {
      const current = prev[id] || "pending";
      let next: "pending" | "investigating" | "cleared" = "pending";
      if (current === "pending") next = "investigating";
      else if (current === "investigating") next = "cleared";
      else next = "pending";
      return { ...prev, [id]: next };
    });
  };

  const getStatusIcon = (status?: string) => {
    if (status === "cleared") return <CheckSquare className="w-5 h-5 text-emerald-500 dark:text-emerald-400" />;
    if (status === "investigating") return <AlertCircle className="w-5 h-5 text-amber-500 dark:text-amber-400" />;
    return <Square className="w-5 h-5 text-slate-300 dark:text-slate-600" />;
  };

  const fetchHistory = useCallback(async () => {
    try {
      const res = await axios.get<HistoryItem[]>(`${API_BASE}/api/history`);
      setHistory(res.data);
    } catch {
      // Silently fail
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatHistory]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(Array.from(e.target.files));
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    files.forEach(f => formData.append("files", f));
    formData.append("industry", industry);

    try {
      const response = await axios.post<AnalysisResult>(
        `${API_BASE}/api/upload`,
        formData
      );
      setResult(response.data);
      setActiveHistoryId(response.data.id ?? null);
      setSidebarOpen(true);
      setChatHistory([{role: 'ai', content: `Hello! I've analyzed the ${industry} financial data. How can I help you investigate?`}]);
      await fetchHistory();
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      if (typeof detail === 'string') {
        setError(detail);
      } else if (Array.isArray(detail)) {
        setError(detail.map((e: any) => e.msg).join(", "));
      } else {
        setError("An error occurred during analysis.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleHistoryClick = async (item: HistoryItem) => {
    if (activeHistoryId === item.id) return;
    setLoadingHistory(true);
    try {
      const res = await axios.get<AnalysisResult>(`${API_BASE}/api/history/${item.id}`);
      setResult(res.data);
      setActiveHistoryId(item.id);
      setStatusMarks({});
      setChatHistory([{role: 'ai', content: `Hello! I've loaded the historic analysis for ${item.filename}. What questions do you have?`}]);
    } catch {
      setError("Failed to load historical record.");
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleNewAudit = () => {
    setResult(null);
    setFiles([]);
    setError(null);
    setActiveHistoryId(null);
    setStatusMarks({});
    setChatOpen(false);
    setChatHistory([]);
  };

  const handleExportPDF = async () => {
    const element = document.getElementById('report-dashboard');
    if (!element) return;
    try {
      const html2pdf = (await import('html2pdf.js')).default;
      const opt = {
        margin:       0.2,
        filename:     `FinAuditAI_Report_${result?.filename || 'audit'}.pdf`,
        image:        { type: 'jpeg', quality: 0.98 },
        html2canvas:  { scale: 2, useCORS: true },
        jsPDF:        { unit: 'in', format: 'letter', orientation: 'landscape' }
      };
      html2pdf().set(opt).from(element).save();
    } catch(e) {
      console.error("PDF export failed", e);
      alert("PDF Export failed. Ensure html2pdf.js is fully loaded.");
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatQuery.trim() || !activeHistoryId) return;
    const userQ = chatQuery;
    setChatHistory(prev => [...prev, {role: "user", content: userQ}]);
    setChatQuery("");
    setChatLoading(true);
    try {
        const res = await axios.post(`${API_BASE}/api/chat/${activeHistoryId}`, { query: userQ });
        setChatHistory(prev => [...prev, {role: "ai", content: res.data.reply}]);
    } catch {
        setChatHistory(prev => [...prev, {role: "ai", content: "Error communicating with AI copilot."}]);
    } finally {
        setChatLoading(false);
    }
  };

  const chartData = (() => {
    if (!result?.ratios) return [];
    const years = result.raw_data_summary.years_analyzed;
    const dataByYear: Record<string, any> = {};
    years.forEach((y) => { dataByYear[y] = { name: y }; });
    result.ratios.forEach((r) => {
      if (dataByYear[r.year]) dataByYear[r.year][r.name] = r.value;
    });
    return Object.values(dataByYear);
  })();

  const Sidebar = () => (
    <aside className={`
      flex-shrink-0 w-80 bg-white dark:bg-[#1a1a1a] border-r border-slate-100 dark:border-white/5
      flex flex-col h-[calc(100vh-4rem)] sticky top-16 transition-transform duration-300 z-30 shadow-2xl lg:shadow-none
      ${sidebarOpen ? "translate-x-0 absolute lg:relative" : "-translate-x-full absolute lg:relative hidden"}
    `}>
      <div className="p-5 border-b border-slate-100 dark:border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="w-5 h-5 text-slate-800 dark:text-slate-300" />
          <span className="font-semibold text-slate-900 dark:text-white">Audit History</span>
        </div>
        <button
          onClick={() => setSidebarOpen(false)}
          className="p-1.5 rounded-full hover:bg-slate-100 dark:hover:bg-white/10 text-slate-500 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="p-4 border-b border-slate-100 dark:border-white/5">
        <button
          onClick={handleNewAudit}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl
            bg-emerald-50 dark:bg-emerald-500/10 hover:bg-emerald-100 dark:hover:bg-emerald-500/20
            text-emerald-700 dark:text-emerald-400 font-semibold transition-all border border-emerald-200 dark:border-emerald-500/20"
        >
          <PlusCircle className="w-5 h-5" />
          Start New Audit
        </button>
      </div>

      <div className="flex-1 overflow-y-auto py-2 custom-scrollbar">
        {history.length === 0 ? (
          <div className="p-8 text-center text-slate-500 text-sm">
            No past audits found.
          </div>
        ) : (
          history.map((item) => (
            <button
              key={item.id}
              onClick={() => handleHistoryClick(item)}
              className={`
                w-full text-left px-5 py-4 border-b border-slate-50 dark:border-white/5 transition-all group
                hover:bg-slate-50 dark:hover:bg-white/5
                ${activeHistoryId === item.id ? "bg-slate-50 dark:bg-white/5 border-l-4 border-l-emerald-500" : "border-l-4 border-l-transparent"}
              `}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`p-2 rounded-lg ${activeHistoryId === item.id ? 'bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400' : 'bg-slate-100 dark:bg-white/10 text-slate-500'}`}>
                    <FileText className="w-4 h-4 flex-shrink-0" />
                  </div>
                  <div className="flex flex-col min-w-0">
                    <span className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
                      {item.filename}
                    </span>
                    <span className="flex items-center gap-1.5 text-xs text-slate-500 mt-1">
                      <Clock className="w-3 h-3" />
                      {timeAgo(item.upload_date)}
                    </span>
                  </div>
                </div>
                <ChevronRight className={`w-4 h-4 flex-shrink-0 transition-transform ${activeHistoryId === item.id ? 'text-emerald-500 translate-x-1' : 'text-slate-300 dark:text-slate-600 group-hover:text-slate-400'}`} />
              </div>
            </button>
          ))
        )}
      </div>
    </aside>
  );

  return (
    <div className="min-h-screen font-sans selection:bg-emerald-500/30 bg-transparent">
      
      {/* ── Header ── */}
      <header className="h-16 bg-white dark:bg-[#1a1a1a] border-b border-slate-100 dark:border-white/5 sticky top-0 z-40 flex items-center justify-between px-6 shadow-sm">
        <div className="flex items-center gap-4">
          {history.length > 0 && !sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 rounded-full hover:bg-slate-100 dark:hover:bg-white/10 text-slate-600 dark:text-slate-400 transition-colors"
            >
              <History className="w-5 h-5" />
            </button>
          )}
          <div className="flex items-center gap-2.5 cursor-pointer" onClick={handleNewAudit}>
            <div className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Activity className="w-4 h-4 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">
              FinAudit<span className="text-emerald-500">AI</span>
            </h1>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {result && (
            <>
              <button 
                onClick={handleExportPDF}
                className="hidden sm:flex items-center gap-2 text-xs font-semibold px-4 py-2 bg-slate-100 dark:bg-white/5 hover:bg-slate-200 dark:hover:bg-white/10 text-slate-700 dark:text-slate-300 rounded-full transition-colors"
              >
                <Download className="w-3.5 h-3.5" /> Export PDF
              </button>
              <span className="hidden md:inline-flex items-center gap-1.5 text-xs px-3 py-1.5 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 rounded-full font-semibold">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Analysis Complete
              </span>
            </>
          )}
          <ThemeToggle />
        </div>
      </header>

      {/* ── Body ── */}
      <div className="flex relative z-10 w-full overflow-hidden">
        <Sidebar />

        <main className="flex-1 min-w-0">
          
          {loadingHistory && (
            <div className="absolute inset-0 bg-white/60 dark:bg-[#121212]/60 backdrop-blur-sm z-20 flex items-center justify-center">
              <div className="w-10 h-10 border-4 border-emerald-500 border-t-transparent rounded-full animate-spin shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
            </div>
          )}

          {/* ── Groww-Style Upload Landing Page ── */}
          {!result && (
            <div className="max-w-6xl mx-auto px-6 py-12 lg:py-24 animate-fade-in flex flex-col lg:flex-row items-center gap-16">
              
              {/* Left Column: Value Prop */}
              <div className="flex-1 text-center lg:text-left">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-semibold text-sm mb-6 border border-emerald-200 dark:border-emerald-500/20 shadow-sm">
                  <ShieldCheck className="w-4 h-4" /> Enterprise-Grade AI Auditor
                </div>
                <h2 className="text-4xl lg:text-6xl font-extrabold text-slate-900 dark:text-white leading-[1.1] mb-6 tracking-tight">
                  Audit Smarter,<br />
                  <span className="text-emerald-500">Not Harder.</span>
                </h2>
                <p className="text-lg text-slate-600 dark:text-slate-400 mb-10 max-w-xl mx-auto lg:mx-0 leading-relaxed">
                  Upload multiple financial statements to instantly cross-reference data, verify GAAP/IFRS compliance, map key ratios against industry benchmarks, and generate professional auditor questions.
                </p>
                
                {/* Floating Mock Stats */}
                <div className="hidden lg:flex gap-6 items-center">
                  <div className="bg-white dark:bg-[#1a1a1a] p-5 rounded-2xl shadow-xl dark:shadow-2xl border border-slate-100 dark:border-white/5 flex items-center gap-4 flex-1 hover:-translate-y-1 transition-transform">
                    <div className="p-3 bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400 rounded-xl">
                      <FileCheck2 className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm text-slate-500 font-medium">Compliance Check</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white mt-0.5">GAAP / IFRS</p>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-[#1a1a1a] p-5 rounded-2xl shadow-xl dark:shadow-2xl border border-slate-100 dark:border-white/5 flex items-center gap-4 flex-1 hover:-translate-y-1 transition-transform">
                    <div className="p-3 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-xl">
                      <MessageSquare className="w-6 h-6" />
                    </div>
                    <div>
                      <p className="text-sm text-slate-500 font-medium">AI Copilot</p>
                      <p className="text-2xl font-bold text-slate-900 dark:text-white mt-0.5">Chat Included</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: Upload Card */}
              <div className="w-full max-w-md">
                <div className="bg-white dark:bg-[#1a1a1a] p-8 rounded-[2rem] shadow-2xl border border-slate-100 dark:border-white/5 relative overflow-hidden group">
                  <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none" />
                  
                  <div className="relative z-10 flex flex-col items-center text-center">
                    <div className="w-20 h-20 bg-emerald-50 dark:bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 text-emerald-500 shadow-inner">
                      <UploadCloud className="w-10 h-10" />
                    </div>
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white mb-2">Upload Statements</h3>
                    
                    <div className="w-full mb-6 text-left">
                      <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 ml-1">Industry Benchmark</label>
                      <select 
                        value={industry}
                        onChange={(e) => setIndustry(e.target.value)}
                        className="w-full bg-slate-50 dark:bg-[#222] border border-slate-200 dark:border-white/10 rounded-xl px-4 py-3 text-sm font-medium focus:ring-2 focus:ring-emerald-500 outline-none transition-all cursor-pointer"
                      >
                        <option value="Technology">Technology (Software/SaaS)</option>
                        <option value="Manufacturing">Manufacturing</option>
                        <option value="Retail">Retail & E-commerce</option>
                        <option value="Healthcare">Healthcare & BioTech</option>
                      </select>
                    </div>

                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      accept=".csv, .xlsx, .xls"
                      multiple
                      onChange={handleFileChange}
                    />
                    <label
                      htmlFor="file-upload"
                      className="cursor-pointer bg-slate-50 dark:bg-black/20 hover:bg-slate-100 dark:hover:bg-black/40 
                        border-2 border-dashed border-slate-200 dark:border-white/10 hover:border-emerald-500 transition-all 
                        rounded-2xl px-6 py-8 w-full flex flex-col items-center gap-3 group/dropzone"
                    >
                      <div className="p-3 bg-white dark:bg-[#222] rounded-full shadow-sm group-hover/dropzone:scale-110 transition-transform">
                        <FileText className="w-6 h-6 text-slate-400 group-hover/dropzone:text-emerald-500" />
                      </div>
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                        {files.length > 0 ? `${files.length} file(s) selected` : "Select multiple files to cross-reference"}
                      </span>
                    </label>

                    <button
                      onClick={handleUpload}
                      disabled={files.length === 0 || loading}
                      className={`mt-6 w-full py-4 rounded-xl font-bold text-lg transition-all flex justify-center items-center gap-2
                        ${files.length === 0 || loading
                          ? "bg-slate-100 dark:bg-white/5 text-slate-400 cursor-not-allowed"
                          : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-[0_8px_25px_-5px_rgba(16,185,129,0.5)] hover:shadow-[0_12px_30px_-5px_rgba(16,185,129,0.6)] hover:-translate-y-0.5"
                        }`}
                    >
                      {loading ? (
                        <>
                          <span className="w-5 h-5 border-2 border-white/40 border-t-white rounded-full animate-spin" />
                          Analyzing Documents...
                        </>
                      ) : (
                        "Run AI Audit"
                      )}
                    </button>
                    {error && (
                      <div className="mt-4 p-3 bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 text-sm font-medium rounded-xl flex items-start gap-2 text-left w-full">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                        <span>{error}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* ── Groww-Style Results Dashboard ── */}
          {result && (
            <div id="report-dashboard" className="animate-fade-in p-6 lg:p-10 max-w-[1400px] mx-auto space-y-8 bg-transparent">
              
              {/* Header */}
              <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div>
                  <h2 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">Audit Report</h2>
                  <div className="flex items-center gap-3 mt-2 text-slate-500 font-medium">
                    <span className="flex items-center gap-1.5"><FileText className="w-4 h-4 text-emerald-500"/> {result.filename || "Unknown"}</span>
                    <span>•</span>
                    <span className="flex items-center gap-1.5"><Clock className="w-4 h-4 text-emerald-500"/> {result.raw_data_summary.years_analyzed.join(", ")}</span>
                  </div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => setChatOpen(!chatOpen)}
                    className="px-5 py-2.5 bg-indigo-500 hover:bg-indigo-600 text-white font-semibold rounded-xl transition-all shadow-md flex items-center gap-2 shadow-indigo-500/30"
                  >
                    <MessageSquare className="w-4 h-4" /> AI Copilot
                  </button>
                  <button
                    onClick={handleNewAudit}
                    className="px-5 py-2.5 bg-white dark:bg-[#1a1a1a] border border-slate-200 dark:border-white/5 hover:border-slate-300 dark:hover:border-white/10 text-slate-700 dark:text-slate-300 font-semibold rounded-xl transition-all shadow-sm flex items-center gap-2"
                  >
                    <X className="w-4 h-4" /> Close
                  </button>
                </div>
              </div>

              {/* Layout Container */}
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start">
                
                {/* Left side: Charts, Compliance & Anomalies Grid (8 cols) */}
                <div className="xl:col-span-8 space-y-8">
                  
                  {/* Compliance Panel */}
                  <div className="bg-white dark:bg-[#1a1a1a] rounded-3xl p-6 shadow-xl dark:shadow-2xl border border-slate-100 dark:border-white/5">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="p-2 bg-blue-50 dark:bg-blue-500/10 rounded-xl">
                        <ShieldCheck className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <h3 className="text-lg font-bold text-slate-900 dark:text-white">GAAP / IFRS Compliance Check</h3>
                    </div>
                    {(!result.compliance_flags || result.compliance_flags.length === 0) ? (
                      <div className="p-4 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 rounded-xl font-medium text-sm flex items-center gap-2">
                        <CheckCircle className="w-5 h-5" /> No glaring formatting or structural violations detected.
                      </div>
                    ) : (
                      <ul className="space-y-2">
                        {result.compliance_flags.map((flag, idx) => (
                          <li key={idx} className="flex items-start gap-3 p-3 bg-amber-50 dark:bg-amber-500/10 text-amber-800 dark:text-amber-300 rounded-xl text-sm font-medium">
                            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-amber-500" />
                            {flag}
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {/* Rich Area Chart with Benchmarks */}
                  <div className="bg-white dark:bg-[#1a1a1a] rounded-3xl p-8 shadow-xl dark:shadow-2xl border border-slate-100 dark:border-white/5">
                    <div className="flex items-center justify-between mb-6">
                      <div>
                        <h3 className="text-xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
                          Financial Metrics Trend
                        </h3>
                        <p className="text-sm text-slate-500 mt-1">Multi-year ratio mapping vs Industry Averages</p>
                      </div>
                      <div className="p-2.5 bg-emerald-50 dark:bg-emerald-500/10 rounded-xl">
                        <TrendingUp className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                      </div>
                    </div>

                    {/* Metric Filter Toggles */}
                    <div className="flex flex-wrap items-center gap-2 mb-6">
                      <button
                        onClick={showAllMetrics}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border ${
                          activeMetrics.size === Object.keys(METRIC_CONFIG).length
                            ? 'bg-slate-900 dark:bg-white text-white dark:text-slate-900 border-slate-900 dark:border-white'
                            : 'bg-white dark:bg-[#222] text-slate-600 dark:text-slate-400 border-slate-200 dark:border-white/10 hover:border-slate-400 dark:hover:border-white/30'
                        }`}
                      >
                        All Metrics
                      </button>
                      {Object.entries(METRIC_CONFIG).map(([metric, cfg]) => (
                        <button
                          key={metric}
                          onClick={() => toggleMetric(metric)}
                          className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all border flex items-center gap-1.5 ${
                            activeMetrics.has(metric)
                              ? 'border-current shadow-sm'
                              : 'bg-white dark:bg-[#222] text-slate-400 dark:text-slate-600 border-slate-200 dark:border-white/10 hover:border-slate-400 dark:hover:border-white/30'
                          }`}
                          style={activeMetrics.has(metric) ? { color: cfg.color, borderColor: cfg.color, backgroundColor: `${cfg.color}10` } : {}}
                        >
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: activeMetrics.has(metric) ? cfg.color : '#cbd5e1' }} />
                          {metric}
                        </button>
                      ))}
                    </div>
                    
                    <div className="h-[350px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                          <defs>
                            <linearGradient id="colorGross" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorCurrent" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorDebt" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#a855f7" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                          <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} dy={10} />
                          <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} axisLine={false} tickLine={false} dx={-10} />
                          <Tooltip
                            contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--tooltip-border)', borderRadius: '12px', padding: '12px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                            itemStyle={{ fontWeight: 600, fontSize: '13px' }}
                            labelStyle={{ color: '#64748b', marginBottom: '4px', fontSize: '12px', fontWeight: 600 }}
                          />
                          <Legend wrapperStyle={{ fontSize: '12px', fontWeight: 500, paddingTop: '20px' }} iconType="circle" />
                          
                          {/* Industry Benchmarks (only show if that metric is active) */}
                          {activeMetrics.has('Current Ratio') && result.industry_benchmarks && result.industry_benchmarks['Current Ratio'] && (
                            <ReferenceLine y={result.industry_benchmarks['Current Ratio']} stroke="#3b82f6" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Avg Current Ratio', fill: '#64748b', fontSize: 10 }} />
                          )}
                          {activeMetrics.has('Gross Margin (%)') && result.industry_benchmarks && result.industry_benchmarks['Gross Margin (%)'] && (
                            <ReferenceLine y={result.industry_benchmarks['Gross Margin (%)']} stroke="#10b981" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Avg Gross Margin', fill: '#64748b', fontSize: 10 }} />
                          )}

                          {/* Conditionally render each Area based on active filter */}
                          {activeMetrics.has('Gross Margin (%)') && (
                            <Area type="monotone" dataKey="Gross Margin (%)" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorGross)" activeDot={{ r: 6, strokeWidth: 0 }} />
                          )}
                          {activeMetrics.has('Current Ratio') && (
                            <Area type="monotone" dataKey="Current Ratio" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorCurrent)" activeDot={{ r: 6, strokeWidth: 0 }} />
                          )}
                          {activeMetrics.has('Debt-to-Equity Ratio') && (
                            <Area type="monotone" dataKey="Debt-to-Equity Ratio" stroke="#f43f5e" strokeWidth={3} fillOpacity={1} fill="url(#colorDebt)" activeDot={{ r: 6, strokeWidth: 0 }} />
                          )}
                          {activeMetrics.has('Net Profit Margin (%)') && (
                            <Area type="monotone" dataKey="Net Profit Margin (%)" stroke="#a855f7" strokeWidth={3} fillOpacity={1} fill="url(#colorProfit)" activeDot={{ r: 6, strokeWidth: 0 }} />
                          )}
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* High Density Table for Anomalies */}
                  <div className="bg-white dark:bg-[#1a1a1a] rounded-3xl shadow-xl dark:shadow-2xl border border-slate-100 dark:border-white/5 overflow-hidden">
                    <div className="px-6 py-5 border-b border-slate-100 dark:border-white/5 bg-slate-50/50 dark:bg-white/[0.02] flex items-center justify-between">
                      <h3 className="text-lg font-bold text-slate-900 dark:text-white flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-full bg-red-50 dark:bg-red-500/10 flex items-center justify-center">
                          <AlertTriangle className="w-4 h-4 text-red-500" />
                        </div>
                        Flagged Anomalies
                      </h3>
                      <span className="text-sm font-semibold px-3 py-1 bg-white dark:bg-[#222] border border-slate-200 dark:border-white/10 rounded-full shadow-sm text-slate-600 dark:text-slate-400">
                        {result.anomalies.length} Records
                      </span>
                    </div>
                    
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm whitespace-nowrap">
                        <thead className="bg-white dark:bg-[#1a1a1a] text-slate-400 dark:text-slate-500 text-xs font-semibold">
                          <tr>
                            <th className="px-6 py-4 border-b border-slate-100 dark:border-white/5 w-12">Action</th>
                            <th className="px-6 py-4 border-b border-slate-100 dark:border-white/5 w-28">Risk Level</th>
                            <th className="px-6 py-4 border-b border-slate-100 dark:border-white/5">Description</th>
                            <th className="px-6 py-4 border-b border-slate-100 dark:border-white/5">Tags</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-50 dark:divide-white/5 bg-white dark:bg-[#1a1a1a]">
                          {result.anomalies.map((anomaly, idx) => {
                            const tickId = `anomaly-${idx}`;
                            return (
                              <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-white/[0.02] transition-colors group">
                                <td className="px-6 py-4">
                                  <button onClick={() => toggleStatus(tickId)} className="focus:outline-none hover:scale-110 transition-transform" title="Toggle Status">
                                    {getStatusIcon(statusMarks[tickId])}
                                  </button>
                                </td>
                                <td className="px-6 py-4">
                                  <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-xs font-bold uppercase border ${severityColor(anomaly.severity)}`}>
                                    {anomaly.severity}
                                  </span>
                                </td>
                                <td className="px-6 py-4 text-slate-700 dark:text-slate-300 font-medium truncate max-w-[300px]" title={anomaly.description}>
                                  {anomaly.description}
                                </td>
                                <td className="px-6 py-4">
                                  <div className="flex gap-2">
                                    {anomaly.related_metrics.slice(0,2).map((m, i) => (
                                      <span key={i} className="px-2.5 py-1 bg-slate-100 dark:bg-white/5 text-slate-600 dark:text-slate-400 rounded-md text-[11px] font-semibold">
                                        {m}
                                      </span>
                                    ))}
                                    {anomaly.related_metrics.length > 2 && <span className="text-[11px] text-slate-500 font-medium">+{anomaly.related_metrics.length - 2}</span>}
                                  </div>
                                </td>
                              </tr>
                            )
                          })}
                          {result.anomalies.length === 0 && (
                            <tr>
                              <td colSpan={4} className="px-6 py-12 text-center text-slate-500 font-medium">
                                <div className="flex flex-col items-center justify-center">
                                  <Check className="w-8 h-8 text-emerald-500 mb-2" />
                                  No anomalies detected in this report.
                                </div>
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>

                {/* Right side: AI Questions Panel (4 cols) */}
                <div className="xl:col-span-4 bg-white dark:bg-[#1a1a1a] border border-slate-100 dark:border-white/5 rounded-3xl shadow-xl dark:shadow-2xl flex flex-col h-[calc(100vh-160px)] sticky top-28">
                  <div className="px-6 py-5 border-b border-slate-100 dark:border-white/5 bg-slate-50/50 dark:bg-white/[0.02] flex items-center gap-3 rounded-t-3xl">
                    <div className="p-2 bg-indigo-50 dark:bg-indigo-500/10 rounded-xl">
                      <BrainCircuit className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-slate-900 dark:text-white">AI Auditor Queries</h3>
                      <p className="text-xs text-slate-500">Auto-generated questions</p>
                    </div>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
                    {result.audit_questions.map((question, idx) => {
                      const qId = `question-${idx}`;
                      const status = statusMarks[qId] || "pending";
                      return (
                        <div key={idx} className={`p-5 rounded-2xl border text-sm transition-all flex gap-4 group cursor-pointer
                          ${status === 'cleared' ? 'bg-slate-50 dark:bg-black/20 border-slate-100 dark:border-white/5 opacity-60' : 'bg-white dark:bg-[#222] border-slate-200 dark:border-white/10 hover:border-emerald-500/50 hover:shadow-md'}
                        `} onClick={() => toggleStatus(qId)}>
                          <div className="mt-0.5 flex-shrink-0 transition-transform group-hover:scale-110">
                            {getStatusIcon(status)}
                          </div>
                          <p className={`font-medium leading-relaxed ${status === 'cleared' ? 'text-slate-400 line-through' : 'text-slate-700 dark:text-slate-200'}`}>
                            {question}
                          </p>
                        </div>
                      )
                    })}
                  </div>
                </div>

              </div>
            </div>
          )}

          {/* ── Chat Widget ── */}
          {chatOpen && result && (
            <div className="fixed bottom-6 right-6 w-96 bg-white dark:bg-[#1a1a1a] rounded-3xl shadow-2xl border border-slate-200 dark:border-white/10 z-50 flex flex-col h-[500px] overflow-hidden animate-fade-in">
              <div className="px-5 py-4 bg-indigo-600 flex justify-between items-center text-white">
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-5 h-5" />
                  <span className="font-bold text-sm">FinAuditAI Copilot</span>
                </div>
                <button onClick={() => setChatOpen(false)} className="hover:bg-indigo-700 p-1 rounded-full transition-colors">
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar bg-slate-50 dark:bg-[#121212]" ref={chatScrollRef}>
                {chatHistory.map((msg, idx) => (
                  <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-[85%] p-3 rounded-2xl text-sm ${
                      msg.role === 'user' 
                        ? 'bg-indigo-600 text-white rounded-br-none' 
                        : 'bg-white dark:bg-[#222] text-slate-800 dark:text-slate-200 border border-slate-200 dark:border-white/10 rounded-bl-none shadow-sm'
                    }`}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="flex justify-start">
                    <div className="max-w-[85%] p-3 rounded-2xl bg-white dark:bg-[#222] border border-slate-200 dark:border-white/10 rounded-bl-none flex gap-1">
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" />
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-75" />
                      <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce delay-150" />
                    </div>
                  </div>
                )}
              </div>
              
              <form onSubmit={handleChatSubmit} className="p-3 bg-white dark:bg-[#1a1a1a] border-t border-slate-200 dark:border-white/10 flex gap-2">
                <input
                  type="text"
                  value={chatQuery}
                  onChange={e => setChatQuery(e.target.value)}
                  placeholder="Ask about the financial data..."
                  className="flex-1 bg-slate-100 dark:bg-[#222] border-none rounded-xl px-4 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none text-slate-900 dark:text-white"
                  disabled={chatLoading}
                />
                <button 
                  type="submit" 
                  disabled={chatLoading || !chatQuery.trim()}
                  className="p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white rounded-xl transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          )}

        </main>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        :root {
          --tooltip-bg: #ffffff;
          --tooltip-border: #f1f5f9;
        }
        .dark {
          --tooltip-bg: #222222;
          --tooltip-border: #333333;
        }
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(10px) }
          to   { opacity: 1; transform: translateY(0) }
        }
        .animate-fade-in { animation: fade-in 0.4s ease-out forwards; }
        .custom-scrollbar::-webkit-scrollbar { width: 6px; height: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 10px; }
        .dark .custom-scrollbar::-webkit-scrollbar-thumb { background: #333333; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }
        .dark .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #444444; }
      `}} />
    </div>
  );
}
