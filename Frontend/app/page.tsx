"use client";

import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type ApiResponse = {
  status: string;
  report: {
    overall: {
      n_rows: number;
      positive_rate: number;
      class_counts: Record<string, number>;
    };
    by_group: Record<
      string,
      { n: number; positive_rate: number; tpr: number | null; fpr: number | null }
    >;
  };
  debug?: any;
};

export default function Home() {
  const [dark, setDark] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [targetCol, setTargetCol] = useState("income");
  const [sensitiveCol, setSensitiveCol] = useState("sex");
  const [predCol, setPredCol] = useState(""); // optional predictions column
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [report, setReport] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [reportLoading, setReportLoading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);

  const uploadDataset = async () => {
    if (!file) {
      alert("Please upload a CSV file");
      return;
    }

    const formData = new FormData();
    formData.append("dataset", file);
    formData.append("target_col", targetCol);
    formData.append("sensitive_col", sensitiveCol);
    if (predCol.trim()) formData.append("predictions_col", predCol.trim());

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/upload-dataset", {
        method: "POST",
        body: formData,
      });
      const data = (await res.json()) as ApiResponse;
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Upload failed!");
    } finally {
      setLoading(false);
    }
  };

  const generateReport = async () => {
    setReportLoading(true);
    setReport("");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/generate-report", {
        method: "POST",
      });
      const data = await res.json();
      if (data?.status === "ok" && typeof data.report === "string") {
        setReport(data.report.replaceAll("\\n", "\n"));
      } else {
        setReport(JSON.stringify(data, null, 2));
      }
    } catch (e) {
      console.error(e);
      setReport("Failed to generate report.");
    } finally {
      setReportLoading(false);
    }
  };

  // NEW: one-click server demo (trains baseline + reweigh & generates LLM report)
  const runServerDemo = async () => {
    setDemoLoading(true);
    setReport("");
    try {
      const res = await fetch("http://127.0.0.1:8000/api/generate-report", {
        method: "POST",
      });
      const data = await res.json();
      if (data?.status === "ok" && typeof data.report === "string") {
        setReport(data.report.replaceAll("\\n", "\n"));
        // Heads-up: for TPR/FPR bars in UI, upload predicted_adult.csv + set predCol="pred"
      } else {
        setReport(JSON.stringify(data, null, 2));
      }
    } catch (e) {
      console.error(e);
      setReport("Failed to run server demo.");
    } finally {
      setDemoLoading(false);
    }
  };

  // NEW: download current JSON result
  const downloadJson = () => {
    if (!result) {
      alert("No results to download yet.");
      return;
    }
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ethosai_fairness_result.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const groupRows = useMemo(() => {
    if (!result?.report?.by_group) return [];
    return Object.entries(result.report.by_group).map(([group, stats]) => ({
      group,
      n: stats.n,
      posRate: stats.positive_rate,
      tpr: stats.tpr,
      fpr: stats.fpr,
    }));
  }, [result]);

  const anyTprOrFpr =
    groupRows.length > 0 && groupRows.some((g) => g.tpr != null || g.fpr != null);

  return (
    <div className={dark ? "dark" : ""}>
      <div className="min-h-screen p-10 bg-gray-100 text-gray-900 dark:bg-slate-900 dark:text-slate-100 transition-colors">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-4xl font-bold">EthosAI – Fairness Analyzer</h1>

          {/* Dark mode toggle */}
          <button
            onClick={() => setDark((d) => !d)}
            className="px-3 py-2 rounded border dark:border-slate-700 dark:hover:bg-slate-800 hover:bg-gray-200"
            aria-label="Toggle dark mode"
            title="Toggle dark mode"
          >
            {dark ? "🌙 Dark" : "☀️ Light"}
          </button>
        </div>

        <div className="bg-white dark:bg-slate-800 p-6 rounded shadow w-full max-w-2xl">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="mb-3"
          />

          <div className="mb-3">
            <label className="font-medium">Target Column:</label>
            <input
              type="text"
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="border dark:border-slate-700 dark:bg-slate-900 p-2 ml-2 rounded"
            />
          </div>

          <div className="mb-3">
            <label className="font-medium">Sensitive Column:</label>
            <input
              type="text"
              value={sensitiveCol}
              onChange={(e) => setSensitiveCol(e.target.value)}
              className="border dark:border-slate-700 dark:bg-slate-900 p-2 ml-2 rounded"
            />
          </div>

          <div className="mb-4">
            <label className="font-medium">Predictions Column (optional):</label>
            <input
              type="text"
              placeholder="e.g., pred"
              value={predCol}
              onChange={(e) => setPredCol(e.target.value)}
              className="border dark:border-slate-700 dark:bg-slate-900 p-2 ml-2 rounded"
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={uploadDataset}
              disabled={loading}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-60"
            >
              {loading ? "Processing..." : "Upload & Analyze"}
            </button>

            <button
              onClick={generateReport}
              disabled={reportLoading}
              className="px-4 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-60"
              title="Generate LLM-written ethics report using your server data"
            >
              {reportLoading ? "Generating..." : "Generate Ethics Report"}
            </button>

            {/* NEW: one-click server demo */}
            <button
              onClick={runServerDemo}
              disabled={demoLoading}
              className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-60"
              title="Run baseline + reweigh training on the server and produce the ethics report"
            >
              {demoLoading ? "Running Demo..." : "End-to-End (Server Demo)"}
            </button>

            {/* NEW: download JSON */}
            <button
              onClick={downloadJson}
              className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-800"
              title="Download current fairness result as JSON"
            >
              Download JSON
            </button>
          </div>

          {!anyTprOrFpr && result && (
            <div className="mt-4 text-sm text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 border border-amber-300 dark:border-amber-700 rounded p-3">
              Hint: TPR/FPR are blank because you didn’t provide predictions. Upload
              <code className="mx-1 px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded">predicted_adult.csv</code>
              and set
              <code className="mx-1 px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded">Predictions Column = pred</code>
              to see TPR/FPR and charts.
            </div>
          )}
        </div>

        {result && (
          <div className="mt-10 bg-white dark:bg-slate-800 p-6 rounded shadow max-w-4xl">
            <h2 className="text-2xl font-bold">Fairness Report</h2>

            <div className="mt-4 text-sm">
              <div className="mb-2">
                <strong>Rows:</strong> {result.report.overall.n_rows} &nbsp;|&nbsp;
                <strong>Positive Rate:</strong>{" "}
                {result.report.overall.positive_rate.toFixed(4)}
              </div>

              {/* KPI strip */}
              {groupRows.length > 0 && (
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {(() => {
                    const female = groupRows.find((g) => g.group === "Female");
                    const male = groupRows.find((g) => g.group === "Male");
                    const prGap =
                      male && female
                        ? Math.abs(male.posRate - female.posRate)
                        : null;
                    const tprGap =
                      male?.tpr != null && female?.tpr != null
                        ? Math.abs((male.tpr ?? 0) - (female.tpr ?? 0))
                        : null;
                    const fprGap =
                      male?.fpr != null && female?.fpr != null
                        ? Math.abs((male.fpr ?? 0) - (female.fpr ?? 0))
                        : null;

                    const Card = ({
                      label,
                      value,
                    }: {
                      label: string;
                      value: number | null;
                    }) => (
                      <div className="rounded-lg border bg-white dark:bg-slate-900 dark:border-slate-700 p-4">
                        <div className="text-xs uppercase tracking-wide text-gray-500 dark:text-slate-400">
                          {label}
                        </div>
                        <div className="text-2xl font-semibold mt-1">
                          {value == null ? "—" : value.toFixed(3)}
                        </div>
                      </div>
                    );

                    return (
                      <>
                        <Card label="Pos-Rate Gap (Male–Female)" value={prGap} />
                        <Card label="TPR Gap (Male–Female)" value={tprGap} />
                        <Card label="FPR Gap (Male–Female)" value={fprGap} />
                      </>
                    );
                  })()}
                </div>
              )}

              {/* Simple table for group stats */}
              <div className="overflow-auto mt-4">
                <table className="w-full text-left border dark:border-slate-700">
                  <thead>
                    <tr className="bg-gray-100 dark:bg-slate-900">
                      <th className="p-2 border dark:border-slate-700">Group</th>
                      <th className="p-2 border dark:border-slate-700">n</th>
                      <th className="p-2 border dark:border-slate-700">Pos Rate</th>
                      <th className="p-2 border dark:border-slate-700">TPR</th>
                      <th className="p-2 border dark:border-slate-700">FPR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {groupRows.map((r) => (
                      <tr key={r.group}>
                        <td className="p-2 border dark:border-slate-700">
                          {r.group}
                        </td>
                        <td className="p-2 border dark:border-slate-700">{r.n}</td>
                        <td className="p-2 border dark:border-slate-700">
                          {r.posRate?.toFixed(3)}
                        </td>
                        <td className="p-2 border dark:border-slate-700">
                          {r.tpr == null ? "—" : r.tpr.toFixed(3)}
                        </td>
                        <td className="p-2 border dark:border-slate-700">
                          {r.fpr == null ? "—" : r.fpr.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Fairness Bar Charts */}
              {groupRows.length > 0 && (
                <div className="mt-8">
                  <h3 className="text-xl font-semibold mb-2">
                    Group Fairness Charts
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* TPR Chart */}
                    <div className="bg-white dark:bg-slate-900 p-4 rounded shadow">
                      <h4 className="font-medium mb-2">True Positive Rate (TPR)</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={groupRows}>
                          <XAxis dataKey="group" />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Bar dataKey="tpr" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* FPR Chart */}
                    <div className="bg-white dark:bg-slate-900 p-4 rounded shadow">
                      <h4 className="font-medium mb-2">False Positive Rate (FPR)</h4>
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={groupRows}>
                          <XAxis dataKey="group" />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Bar dataKey="fpr" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>
              )}

              {/* Raw JSON (keep for debugging) */}
              <pre className="mt-4 bg-gray-900 text-green-300 p-4 rounded overflow-auto">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {report && (
          <div className="mt-10 bg-white dark:bg-slate-800 p-6 rounded shadow max-w-4xl">
            <h2 className="text-2xl font-bold">Ethics Report (LLM)</h2>
            <pre className="mt-4 whitespace-pre-wrap bg-gray-900 text-green-300 p-4 rounded overflow-auto">
              {report}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
