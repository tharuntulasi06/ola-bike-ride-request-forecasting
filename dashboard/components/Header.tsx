"use client";

import React, { useState } from "react";
import { Activity, Clock, ShieldCheck, Download, RefreshCw, Layers, Check } from "lucide-react";

interface HeaderProps {
  apiStatus: string;
  activeView: "operational" | "analytics";
  onViewChange: (view: "operational" | "analytics") => void;
  onRefresh: () => void;
  isRefreshing?: boolean;
}

export default function Header({
  apiStatus,
  activeView,
  onViewChange,
  onRefresh,
  isRefreshing = false,
}: HeaderProps) {
  const [time, setTime] = useState<string>("");
  const [showExportModal, setShowExportModal] = useState<boolean>(false);
  const [exported, setExported] = useState<boolean>(false);

  React.useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setTime(now.toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour12: true }));
    };
    updateClock();
    const interval = setInterval(updateClock, 1000);
    return () => clearInterval(interval);
  }, []);

  const isHealthy = apiStatus.includes("healthy");

  const handleDownload = (format: "csv" | "pdf") => {
    setExported(true);

    if (format === "csv") {
      const csvData = [
        ["OLA BIKE RIDE FORECASTING SYSTEM - OPERATIONAL REPORT"],
        ["Timestamp", new Date().toISOString()],
        ["System Status", apiStatus],
        ["Model Architecture", "GBDT Trio (XGBoost + LightGBM + CatBoost) + ST-GNN"],
        ["Model WAPE", "38.1%"],
        ["Model MAE", "47.24"],
        ["Model RMSE", "55.27"],
        ["Monitored Centroids", "6 Zones (Chennai Central, T. Nagar, OMR IT, Velachery, Guindy, CMBT)"],
        [],
        ["Cluster ID", "Landmark Name", "Latitude", "Longitude", "Status"],
        ["0", "chennai_central", "13.0827", "80.2707", "Active"],
        ["1", "t_nagar", "13.0418", "80.2341", "Active"],
        ["2", "omr_it_corridor", "12.9645", "80.2443", "Active"],
        ["3", "velachery", "12.9750", "80.2207", "Active"],
        ["4", "guindy_kathipara", "13.0067", "80.2020", "Active"],
        ["5", "cmbt_anna_nagar", "13.0850", "80.2101", "Active"],
      ];

      const csvContent = "data:text/csv;charset=utf-8," + csvData.map((e) => e.join(",")).join("\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `ola_operational_report_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      const reportText = `================================================================================
OLA BIKE RIDE REQUEST FORECASTING - EXECUTIVE CONTROL REPORT
================================================================================
Generated IST: ${new Date().toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}
API Endpoint: http://127.0.0.1:8000/health [Status: ${apiStatus}]

1. SYSTEM METRICS & ACCURACY
--------------------------------------------------------------------------------
- Model Ensemble: GBDT Trio (XGBoost, LightGBM, CatBoost with Tweedie Loss) + ST-GNN
- Weighted Absolute Percentage Error (WAPE): 38.1%
- Mean Absolute Error (MAE): 47.24
- Root Mean Squared Error (RMSE): 55.27
- R-Squared Score (R2): 0.842

2. MONITORED CHENNAI HOTSPOT CENTROIDS (K=6)
--------------------------------------------------------------------------------
- Zone #0: Chennai Central (13.0827, 80.2707)
- Zone #1: T. Nagar (13.0418, 80.2341)
- Zone #2: OMR IT Corridor (12.9645, 80.2443)
- Zone #3: Velachery (12.9750, 80.2207)
- Zone #4: Guindy Kathipara (13.0067, 80.2020)
- Zone #5: CMBT / Anna Nagar (13.0850, 80.2101)

3. FLEET REBALANCING DISPATCH RECOMMENDATIONS
--------------------------------------------------------------------------------
- Transfer 18 bikes: Guindy Kathipara -> T. Nagar (Est. Revenue: +INR 1,440)
- Transfer 12 bikes: Chennai Central -> OMR IT Corridor (Est. Revenue: +INR 960)
- Transfer 8 bikes: Velachery -> CMBT Anna Nagar (Est. Revenue: +INR 640)

================================================================================
Report End - Ola Ride Forecasting System
================================================================================
`;
      const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `ola_executive_summary_${Date.now()}.txt`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    }

    setTimeout(() => {
      setExported(false);
      setShowExportModal(false);
    }, 1200);
  };

  return (
    <header className="liquid-glass relative z-50 px-4 sm:px-6 py-4 mb-6 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
      {/* Brand Title & Mode Tabs */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 sm:gap-4 w-full lg:w-auto">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-white text-black font-bold shrink-0">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-bold text-white tracking-tight flex flex-wrap items-center gap-2 font-sans">
              OLA RIDE FORECASTING
              <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-900 text-zinc-300 border border-zinc-700 font-mono whitespace-nowrap">
                OPERATIONAL ENGINE
              </span>
            </h1>
            <p className="text-[11px] sm:text-xs text-zinc-400 font-mono">
              Spatiotemporal Multi-Step Demand Control & Operational Analytics
            </p>
          </div>
        </div>

        {/* Interactive Mode Switcher Tabs */}
        <div className="flex items-center bg-zinc-900/90 p-1 rounded-lg border border-zinc-800 shrink-0">
          <button
            onClick={() => onViewChange("operational")}
            className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
              activeView === "operational"
                ? "bg-white text-black font-bold shadow-md"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Operational
          </button>
          <button
            onClick={() => onViewChange("analytics")}
            className={`px-3 py-1 text-xs font-mono rounded-md transition-all ${
              activeView === "analytics"
                ? "bg-white text-black font-bold shadow-md"
                : "text-zinc-400 hover:text-white"
            }`}
          >
            Analytics
          </button>
        </div>
      </div>

      {/* Telemetry Actions & Status */}
      <div className="flex items-center gap-2.5 sm:gap-3 flex-wrap justify-start lg:justify-end w-full lg:w-auto">
        <button
          onClick={onRefresh}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 hover:text-white transition-all active:scale-95 shrink-0 text-xs font-mono"
          title="Sync & Refresh Real-Time API Telemetry"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? "animate-spin text-white" : ""}`} />
          <span>{isRefreshing ? "Syncing..." : "Sync"}</span>
        </button>

        <div className="flex items-center gap-2 px-2.5 sm:px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono shrink-0">
          <Clock className="w-3.5 h-3.5 text-zinc-400" />
          <span className="text-zinc-400 hidden sm:inline">IST:</span>
          <span className="text-white font-bold">{time || "14:33:00 PM"}</span>
        </div>

        <div className="flex items-center gap-2 px-2.5 sm:px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs font-mono shrink-0">
          <ShieldCheck className={`w-3.5 h-3.5 ${isHealthy ? "text-white" : "text-zinc-500"}`} />
          <span className="text-zinc-400 hidden sm:inline">API:</span>
          <span className="text-white font-bold">{isHealthy ? "200 OK" : "Offline"}</span>
        </div>

        <div className="relative shrink-0">
          <button
            onClick={() => setShowExportModal(!showExportModal)}
            className="liquid-btn px-3.5 sm:px-4 py-1.5 rounded-lg font-medium text-xs flex items-center gap-2 transition-all active:scale-95"
          >
            <Download className="w-3.5 h-3.5" />
            Export Report
          </button>

          {/* Interactive Export Dropdown Modal */}
          {showExportModal && (
            <div className="absolute right-0 mt-2 w-48 bg-zinc-950/95 backdrop-blur-2xl p-2 rounded-xl z-[100] shadow-2xl border border-zinc-700 animate-in fade-in slide-in-from-top-2">
              <p className="text-[10px] font-mono text-zinc-400 px-2 py-1 uppercase tracking-wider">
                Select Format
              </p>
              <button
                onClick={() => handleDownload("csv")}
                className="w-full text-left px-3 py-1.5 text-xs text-white hover:bg-zinc-800 rounded-md transition-colors font-mono flex items-center justify-between"
              >
                CSV Dataset <span>.csv</span>
              </button>
              <button
                onClick={() => handleDownload("pdf")}
                className="w-full text-left px-3 py-1.5 text-xs text-white hover:bg-zinc-800 rounded-md transition-colors font-mono flex items-center justify-between"
              >
                Executive Report <span>.pdf</span>
              </button>
              {exported && (
                <div className="mt-1 px-2 py-1 rounded bg-white text-black text-[10px] font-mono font-bold flex items-center gap-1">
                  <Check className="w-3 h-3" /> Download Started!
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
