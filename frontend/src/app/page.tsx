"use client";

import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import {
  UploadCloud, FileText, AlertTriangle, CheckCircle,
  BrainCircuit, Activity, BarChart3, History, X, Clock,
  ChevronRight, PlusCircle, TrendingUp, ShieldAlert
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import { GradientText } from "@/components/ui/gradient-text";
import { ThemeToggle } from "@/components/ui/theme-toggle";

const API_BASE = "http://localhost:8000";

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
  if (severity === "High") return "text-red-400 bg-red-500/20 border-red-500/30";
  if (severity === "Medium") return "text-amber-400 bg-amber-500/20 border-amber-500/30";
  return "text-emerald-400 bg-emerald-500/20 border-emerald-500/30";
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState<number | null>(null);

  // Fetch sidebar history list
  const fetchHistory = useCallback(async () => {
    try {
      const res = await axios.get<HistoryItem[]>(`${API_BASE}/api/history`);
      setHistory(res.data);
    } catch {
      // Silently fail — history is non-critical
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post<AnalysisResult>(
        `${API_BASE}/api/upload`,
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );
      setResult(response.data);
      setActiveHistoryId(response.data.id ?? null);
      setSidebarOpen(true);
      await fetchHistory(); // Refresh sidebar list
    } catch (err: any) {
      setError(err.response?.data?.detail || "An error occurred during analysis.");
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
    } catch {
      setError("Failed to load historical record.");
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleNewAudit = () => {
    setResult(null);
    setFile(null);
    setError(null);
    setActiveHistoryId(null);
  };

  // Chart data derived from result
  const prepareChartData = () => {
    if (!result?.ratios) return [];
    const years = result.raw_data_summary.years_analyzed;
    const dataByYear: Record<string, any> = {};
    years.forEach((y) => { dataByYear[y] = { name: y }; });
    result.ratios.forEach((r) => {
      if (dataByYear[r.year]) dataByYear[r.year][r.name] = r.value;
    });
    return Object.values(dataByYear);
  };

  const chartData = prepareChartData();
  const latestYear = result?.raw_data_summary.years_analyzed.at(-1);
  const latestRatios = result?.ratios.filter((r) => r.year === latestYear) ?? [];

  // ── Sidebar ────────────────────────────────────────────────────────────────

  const Sidebar = () => (
    <aside className={`
      flex-shrink-0 w-72 bg-slate-900/70 border-r border-white/10 backdrop-blur-xl
      flex flex-col h-screen sticky top-0 transition-all duration-300
      ${sidebarOpen ? "translate-x-0" : "-translate-x-full hidden"}
    `}>
      {/* Sidebar header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <History className="w-4 h-4 text-indigo-400" />
          <span className="text-sm font-semibold text-white">Audit History</span>
        </div>
        <button
          onClick={() => setSidebarOpen(false)}
          className="p-1 rounded-md hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
          title="Close sidebar"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* New Audit Button */}
      <div className="p-3 border-b border-white/5">
        <button
          onClick={handleNewAudit}
          className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg
            bg-indigo-600/20 hover:bg-indigo-600/30 border border-indigo-500/30
            text-indigo-300 hover:text-indigo-200 text-sm font-medium transition-all"
        >
          <PlusCircle className="w-4 h-4" />
          New Audit
        </button>
      </div>

      {/* History list */}
      <div className="flex-1 overflow-y-auto py-2 custom-scrollbar">
        {history.length === 0 ? (
          <div className="p-6 text-center text-slate-500 text-sm">
            No past audits yet.
          </div>
        ) : (
          history.map((item) => (
            <button
              key={item.id}
              onClick={() => handleHistoryClick(item)}
              className={`
                w-full text-left px-4 py-3 border-b border-white/5 transition-all group
                hover:bg-white/5
                ${activeHistoryId === item.id ? "bg-indigo-500/10 border-l-2 border-l-indigo-500" : "border-l-2 border-l-transparent"}
              `}
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="w-3.5 h-3.5 flex-shrink-0 text-slate-400" />
                  <span className="text-xs font-medium text-slate-200 truncate">
                    {item.filename}
                  </span>
                </div>
                <ChevronRight className="w-3 h-3 flex-shrink-0 text-slate-600 group-hover:text-slate-400 mt-0.5" />
              </div>
              <div className="flex items-center gap-3 mt-1.5 pl-5">
                <span className="flex items-center gap-1 text-[10px] text-slate-500">
                  <Clock className="w-2.5 h-2.5" />
                  {timeAgo(item.upload_date)}
                </span>
                {item.anomaly_count > 0 && (
                  <span className="flex items-center gap-1 text-[10px] text-red-400/80">
                    <ShieldAlert className="w-2.5 h-2.5" />
                    {item.anomaly_count} anomal{item.anomaly_count === 1 ? "y" : "ies"}
                  </span>
                )}
                <span className="flex items-center gap-1 text-[10px] text-blue-400/80">
                  <TrendingUp className="w-2.5 h-2.5" />
                  {item.ratio_count} ratio{item.ratio_count !== 1 ? "s" : ""}
                </span>
              </div>
            </button>
          ))
        )}
      </div>
    </aside>
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen text-black dark:text-white font-serif selection:bg-indigo-500/30 bg-transparent">
      {/* Background noise texture */}
      <div className="fixed inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-soft-light pointer-events-none z-0" />

      {/* ── Header ── */}
      <header className="border-b border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-full px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Sidebar toggle — only show when there is history */}
            {history.length > 0 && !sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                className="p-2 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition-colors"
                title="Show history"
              >
                <History className="w-4 h-4" />
              </button>
            )}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <h1 className="text-xl font-bold tracking-tight text-black dark:text-white">
                FinAudit<GradientText>AI</GradientText>
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm font-medium text-slate-500 dark:text-slate-400">
            {result && (
              <span className="text-xs px-2 py-1 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 rounded-full">
                Analysis Active
              </span>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* ── Body: Sidebar + Main ── */}
      <div className="flex relative z-10">
        <Sidebar />

        <main className="flex-1 min-w-0">

          {/* Loading overlay when switching history */}
          {loadingHistory && (
            <div className="absolute inset-0 bg-white/60 dark:bg-black/60 backdrop-blur-sm z-20 flex items-center justify-center">
              <div className="flex flex-col items-center gap-3">
                <div className="w-8 h-8 border-2 border-indigo-500 dark:border-indigo-400 border-t-transparent rounded-full animate-spin" />
                <span className="text-sm text-slate-500 dark:text-slate-400">Loading audit record…</span>
              </div>
            </div>
          )}

          {/* ── Upload Screen ── */}
          {!result && (
            <div className="max-w-3xl mx-auto text-center mt-20 px-6 animate-fade-in-up">
              <h2 className="text-5xl font-extrabold tracking-tight mb-6">
                <GradientText>Intelligent</GradientText> Audit Preparation
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400 mb-12 max-w-2xl mx-auto leading-relaxed">
                Upload your financial statements and let our AI agent instantly detect anomalies,
                calculate key ratios, and generate professional auditor questions.
              </p>

              <div className="bg-black/5 dark:bg-white/5 backdrop-blur-xl border border-black/10 dark:border-white/10 p-10 rounded-3xl shadow-2xl transition-all hover:border-black/20 dark:hover:border-white/20">
                <div className="flex flex-col items-center justify-center">
                  <div className="w-20 h-20 bg-indigo-500/10 rounded-full flex items-center justify-center mb-6">
                    <UploadCloud className="w-10 h-10 text-indigo-400" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">Upload Financial Statement</h3>
                  <p className="text-slate-500 mb-8 text-sm">Supports Excel (.xlsx, .xls) and CSV files.</p>

                  <input
                    type="file"
                    id="file-upload"
                    className="hidden"
                    accept=".csv, .xlsx, .xls"
                    onChange={handleFileChange}
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer bg-white/5 hover:bg-white/10 border border-dashed border-slate-600
                      hover:border-indigo-400 transition-colors rounded-xl px-12 py-8 w-full flex flex-col items-center gap-3"
                  >
                    <FileText className="w-6 h-6 text-slate-400" />
                    <span className="text-slate-300 font-medium">
                      {file ? file.name : "Select a file or drag and drop"}
                    </span>
                  </label>

                  <button
                    onClick={handleUpload}
                    disabled={!file || loading}
                    className={`mt-8 px-8 py-3 rounded-xl font-semibold text-white transition-all
                      shadow-[0_0_20px_-5px_rgba(99,102,241,0.5)] flex items-center gap-2
                      ${!file || loading
                        ? "bg-slate-800 text-slate-500 cursor-not-allowed shadow-none"
                        : "bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 hover:scale-105"
                      }`}
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analyzing Data…
                      </span>
                    ) : (
                      <>
                        <BrainCircuit className="w-5 h-5" />
                        Run AI Analysis
                      </>
                    )}
                  </button>
                  {error && <p className="text-red-400 mt-4 text-sm font-medium">{error}</p>}
                </div>
              </div>
            </div>
          )}

          {/* ── Results Dashboard ── */}
          {result && (
            <div className="animate-fade-in px-6 py-8 space-y-8">
              {/* Report header */}
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-1">Audit Analysis Report</h2>
                  <p className="text-slate-400 text-sm">
                    {result.filename && <span className="mr-2 font-medium text-slate-300">{result.filename}</span>}
                    Years: {result.raw_data_summary.years_analyzed.join(", ")}
                  </p>
                </div>
                <button
                  onClick={handleNewAudit}
                  className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-sm font-medium rounded-lg
                    transition-colors border border-white/5 flex items-center gap-2"
                >
                  <PlusCircle className="w-4 h-4" />
                  New Audit
                </button>
              </div>

              {/* Quick stats row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Years Analyzed", value: result.raw_data_summary.years_analyzed.length, icon: <Activity className="w-4 h-4" />, color: "indigo" },
                  { label: "Ratios Computed", value: result.ratios.length, icon: <BarChart3 className="w-4 h-4" />, color: "blue" },
                  { label: "Anomalies Found", value: result.anomalies.length, icon: <AlertTriangle className="w-4 h-4" />, color: result.anomalies.length > 0 ? "red" : "emerald" },
                  { label: "AI Questions", value: result.audit_questions.length, icon: <BrainCircuit className="w-4 h-4" />, color: "purple" },
                ].map((stat, i) => (
                  <div key={i} className="bg-slate-900/40 border border-white/10 rounded-xl p-4 flex items-center gap-3">
                    <div className={`p-2 rounded-lg bg-${stat.color}-500/10 text-${stat.color}-400`}>
                      {stat.icon}
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-white">{stat.value}</p>
                      <p className="text-xs text-slate-400">{stat.label}</p>
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

                {/* Ratios chart + latest values */}
                <div className="col-span-1 lg:col-span-2 bg-slate-900/40 border border-white/10 rounded-2xl p-6 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-blue-500/10 rounded-lg">
                      <BarChart3 className="w-5 h-5 text-blue-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white">Financial Ratios Trend</h3>
                  </div>

                  <div className="h-72 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" />
                        <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                        <YAxis stroke="#94a3b8" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.5rem', color: '#f8fafc' }}
                          itemStyle={{ color: '#cbd5e1' }}
                        />
                        <Legend wrapperStyle={{ fontSize: '11px', color: '#94a3b8' }} />
                        <Line type="monotone" dataKey="Gross Margin (%)" stroke="#8b5cf6" strokeWidth={2.5} dot={{ r: 4 }} activeDot={{ r: 7 }} />
                        <Line type="monotone" dataKey="Current Ratio" stroke="#3b82f6" strokeWidth={2.5} dot={{ r: 4 }} />
                        <Line type="monotone" dataKey="Debt-to-Equity Ratio" stroke="#ec4899" strokeWidth={2.5} dot={{ r: 4 }} />
                        <Line type="monotone" dataKey="Net Profit Margin (%)" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {latestRatios.length > 0 && (
                    <div className="mt-5 grid grid-cols-2 md:grid-cols-3 gap-3">
                      {latestRatios.map((ratio, idx) => (
                        <div key={idx} className="bg-slate-800/50 p-4 rounded-xl border border-white/5">
                          <p className="text-[10px] text-slate-400 mb-1 uppercase tracking-wide">{ratio.name}</p>
                          <p className="text-xl font-bold text-white">{ratio.value}</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">{latestYear}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Anomalies panel */}
                <div className="col-span-1 bg-slate-900/40 border border-white/10 rounded-2xl p-6 backdrop-blur-md flex flex-col">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-red-500/10 rounded-lg">
                      <AlertTriangle className="w-5 h-5 text-red-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-white">Detected Anomalies</h3>
                  </div>

                  <div className="flex-1 overflow-y-auto space-y-4 pr-1 custom-scrollbar">
                    {result.anomalies.length > 0 ? (
                      result.anomalies.map((anomaly, idx) => (
                        <div key={idx} className="p-4 rounded-xl bg-red-500/5 border border-red-500/20">
                          <div className="mb-2">
                            <span className={`text-[10px] font-bold uppercase px-2 py-1 rounded-md border ${severityColor(anomaly.severity)}`}>
                              {anomaly.severity} Risk
                            </span>
                          </div>
                          <p className="text-sm text-slate-300 leading-relaxed">{anomaly.description}</p>
                          <div className="mt-3 flex flex-wrap gap-2">
                            {anomaly.related_metrics.map((metric, mIdx) => (
                              <span key={mIdx} className="text-[10px] px-2 py-0.5 bg-slate-800 text-slate-400 rounded-full border border-slate-700">
                                {metric}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full text-center p-6 border border-dashed border-emerald-500/20 rounded-xl bg-emerald-500/5">
                        <CheckCircle className="w-10 h-10 text-emerald-400 mb-3" />
                        <p className="text-emerald-300 font-medium">No significant anomalies detected.</p>
                        <p className="text-xs text-emerald-400/70 mt-2">Financial trends appear consistent.</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* AI Questions panel */}
                <div className="col-span-1 lg:col-span-3 bg-gradient-to-br from-indigo-900/20 to-purple-900/20 border border-indigo-500/20 rounded-2xl p-8 backdrop-blur-md">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-indigo-500/20 rounded-lg">
                      <BrainCircuit className="w-6 h-6 text-indigo-400" />
                    </div>
                    <h3 className="text-2xl font-bold text-white">AI Auditor Questions</h3>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {result.audit_questions.map((question, idx) => (
                      <div
                        key={idx}
                        className="flex items-start gap-4 p-5 rounded-xl bg-slate-900/60 border border-white/5
                          transition-colors hover:border-indigo-500/30 group"
                      >
                        <div className="w-8 h-8 rounded-full bg-indigo-500/10 flex items-center justify-center
                          flex-shrink-0 text-indigo-400 font-bold border border-indigo-500/20 group-hover:bg-indigo-500/20 text-sm">
                          {idx + 1}
                        </div>
                        <p className="text-slate-200 leading-relaxed pt-0.5 text-sm">{question}</p>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            </div>
          )}
        </main>
      </div>

      {/* Global styles */}
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(20px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fade-in {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        .animate-fade-in-up { animation: fade-in-up 0.6s ease-out forwards; }
        .animate-fade-in    { animation: fade-in 0.5s ease-out forwards; }
        .custom-scrollbar::-webkit-scrollbar       { width: 5px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(15,23,42,0.5); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(51,65,85,0.8); border-radius: 10px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(71,85,105,1); }
      `}} />
    </div>
  );
}
