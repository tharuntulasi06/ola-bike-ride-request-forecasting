"use client";

import React, { useState } from "react";
import { Zap, TrendingUp, Bike, MapPin, Check } from "lucide-react";
import { EvaluationMetrics } from "../types";

interface KPICardsProps {
  metrics: EvaluationMetrics | null;
  activeClusterName: string;
  activeKpiFilter: string | null;
  onSelectKpiFilter: (filterName: string | null) => void;
  actualSurge?: number;
  predictedSurge?: number;
}

export default function KPICards({
  metrics,
  activeClusterName,
  activeKpiFilter,
  onSelectKpiFilter,
  actualSurge = 1.15,
  predictedSurge = 1.45,
}: KPICardsProps) {
  const h1 = metrics?.horizons?.[0];
  const rawWape = h1 ? h1.wape * 100 : (metrics?.metrics?.WAPE ?? 38.1);
  const rawMae = h1 ? h1.mae : (metrics?.metrics?.MAE ?? 47.24);
  const rawRmse = h1 ? h1.rmse : (metrics?.metrics?.RMSE ?? 55.27);

  const wape = rawWape.toFixed(1);
  const mae = rawMae.toFixed(1);
  const rmse = rawRmse.toFixed(1);

  const handleCardClick = (id: string) => {
    if (activeKpiFilter === id) {
      onSelectKpiFilter(null);
    } else {
      onSelectKpiFilter(id);
    }
  };

  const surgeDelta = (predictedSurge - actualSurge).toFixed(2);
  const surgeSign = predictedSurge >= actualSurge ? "+" : "";

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* KPI 1: Active Fleet */}
      <div
        onClick={() => handleCardClick("fleet")}
        className={`liquid-glass p-4 cursor-pointer transition-all ${
          activeKpiFilter === "fleet" ? "liquid-glass-active ring-1 ring-white" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-zinc-400 font-mono uppercase tracking-wider">Total Active Fleet</p>
            <h3 className="text-2xl font-bold text-white font-mono mt-1">
              1,250 <span className="text-xs text-zinc-400 font-normal">Bikes</span>
            </h3>
            <p className="text-[10px] text-zinc-400 mt-1 flex items-center gap-1 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-white animate-ping"></span>
              100% Operational Readiness
            </p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 text-white border border-zinc-800">
            <Bike className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* KPI 2: Actual vs Predicted Surge Multipliers */}
      <div
        onClick={() => handleCardClick("surge")}
        className={`liquid-glass p-4 cursor-pointer transition-all ${
          activeKpiFilter === "surge" ? "liquid-glass-active ring-1 ring-white" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-zinc-400 font-mono uppercase tracking-wider">Actual vs Predicted Surge</p>
            <div className="flex items-baseline gap-2 mt-1 font-mono">
              <div className="flex flex-col">
                <span className="text-[9px] text-zinc-500 uppercase">Actual</span>
                <span className="text-lg font-bold text-zinc-400">{actualSurge.toFixed(2)}x</span>
              </div>
              <span className="text-zinc-600 text-sm">→</span>
              <div className="flex flex-col">
                <span className="text-[9px] text-zinc-400 uppercase">Predicted</span>
                <span className="text-2xl font-bold text-white">{predictedSurge.toFixed(2)}x</span>
              </div>
            </div>
            <p className="text-[10px] text-zinc-400 mt-1 font-mono">
              Delta: <span className="text-white font-bold">{surgeSign}{surgeDelta}x</span> ({activeClusterName.replace(/_/g, " ").toUpperCase()})
            </p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 text-white border border-zinc-800 shrink-0">
            <Zap className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* KPI 3: Model WAPE */}
      <div
        onClick={() => handleCardClick("wape")}
        className={`liquid-glass p-4 cursor-pointer transition-all ${
          activeKpiFilter === "wape" ? "liquid-glass-active ring-1 ring-white" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-zinc-400 font-mono uppercase tracking-wider">Model WAPE</p>
            <h3 className="text-2xl font-bold text-white font-mono mt-1">{wape}%</h3>
            <p className="text-[10px] text-zinc-400 mt-1 font-mono">
              MAE: {mae} | RMSE: {rmse}
            </p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 text-white border border-zinc-800">
            <TrendingUp className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* KPI 4: Monitored Hotspots */}
      <div
        onClick={() => handleCardClick("hotspots")}
        className={`liquid-glass p-4 cursor-pointer transition-all ${
          activeKpiFilter === "hotspots" ? "liquid-glass-active ring-1 ring-white" : ""
        }`}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-zinc-400 font-mono uppercase tracking-wider">Monitored Centroids</p>
            <h3 className="text-2xl font-bold text-white font-mono mt-1">6 Zones</h3>
            <p className="text-[10px] text-zinc-400 mt-1 font-mono">
              Chennai Spatial Grid Matrix
            </p>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 text-white border border-zinc-800">
            <MapPin className="w-5 h-5" />
          </div>
        </div>
      </div>
    </div>
  );
}
