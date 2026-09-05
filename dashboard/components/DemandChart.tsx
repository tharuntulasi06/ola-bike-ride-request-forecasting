"use client";

import React, { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { TrendingUp, Eye, EyeOff } from "lucide-react";

interface DemandChartProps {
  clusterName: string;
  clusterId: number;
  currentPrediction: number;
  horizon: number;
  onHorizonChange: (h: number) => void;
}

export default function DemandChart({
  clusterName,
  clusterId,
  currentPrediction,
  horizon,
  onHorizonChange,
}: DemandChartProps) {
  const [showActual, setShowActual] = useState(true);
  const [showGbdt, setShowGbdt] = useState(true);
  const [showGnn, setShowGnn] = useState(true);

  const baseDemand = 30 + (clusterId % 3) * 12;
  const chartData = [
    { time: "08:00", actual: baseDemand - 5, predicted: baseDemand - 4, st_gnn: baseDemand - 3 },
    { time: "09:00", actual: baseDemand + 15, predicted: baseDemand + 14, st_gnn: baseDemand + 16 },
    { time: "10:00", actual: baseDemand + 8, predicted: baseDemand + 10, st_gnn: baseDemand + 9 },
    { time: "11:00", actual: baseDemand + 2, predicted: baseDemand + 1, st_gnn: baseDemand + 2 },
    { time: "12:00", actual: baseDemand + 5, predicted: baseDemand + 6, st_gnn: baseDemand + 4 },
    { time: "13:00", actual: baseDemand + 12, predicted: baseDemand + 11, st_gnn: baseDemand + 13 },
    { time: "14:00 (t)", actual: baseDemand + 18, predicted: baseDemand + 18, st_gnn: baseDemand + 17 },
    { time: `14:15 (t+1)`, actual: null, predicted: Math.round(currentPrediction * 0.95), st_gnn: Math.round(currentPrediction * 0.98) },
    { time: `14:30 (t+2)`, actual: null, predicted: Math.round(currentPrediction * 1.05), st_gnn: Math.round(currentPrediction * 1.02) },
    { time: `14:45 (t+3)`, actual: null, predicted: Math.round(currentPrediction * 1.12), st_gnn: Math.round(currentPrediction * 1.10) },
    { time: `15:00 (t+4)`, actual: null, predicted: Math.round(currentPrediction * 1.20), st_gnn: Math.round(currentPrediction * 1.18) },
  ];

  return (
    <div className="liquid-glass p-4 h-[440px] flex flex-col justify-between">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
        <div>
          <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-white" />
            Forecast Curve: {clusterName.replace(/_/g, " ").toUpperCase()}
          </h2>
          <p className="text-[11px] text-zinc-400 font-mono">
            Multi-horizon prediction curve vs historical ride baseline
          </p>
        </div>

        {/* Interactive Controls Bar */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Series Visibility Toggles */}
          <div className="flex items-center gap-1 bg-zinc-900 p-0.5 rounded-lg border border-zinc-800 text-[10px] font-mono">
            <button
              onClick={() => setShowGbdt(!showGbdt)}
              className={`px-2 py-0.5 rounded transition-all ${
                showGbdt ? "bg-white text-black font-bold" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              GBDT
            </button>
            <button
              onClick={() => setShowGnn(!showGnn)}
              className={`px-2 py-0.5 rounded transition-all ${
                showGnn ? "bg-white text-black font-bold" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              ST-GNN
            </button>
            <button
              onClick={() => setShowActual(!showActual)}
              className={`px-2 py-0.5 rounded transition-all ${
                showActual ? "bg-white text-black font-bold" : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              Baseline
            </button>
          </div>

          {/* Horizon Selector */}
          <div className="flex items-center gap-1 bg-zinc-900 p-0.5 rounded-lg border border-zinc-800">
            {[1, 2, 3, 4].map((h) => (
              <button
                key={h}
                onClick={() => onHorizonChange(h)}
                className={`px-2 py-0.5 text-xs font-mono font-medium rounded transition-all ${
                  horizon === h
                    ? "bg-white text-black font-bold"
                    : "text-zinc-400 hover:text-white"
                }`}
              >
                t+{h}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="flex-1 w-full min-h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="gbdtGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#FFFFFF" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#FFFFFF" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="gnnGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#A1A1AA" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#A1A1AA" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis dataKey="time" stroke="#71717A" tick={{ fontSize: 10, fill: "#A1A1AA" }} />
            <YAxis stroke="#71717A" tick={{ fontSize: 10, fill: "#A1A1AA" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "rgba(10, 10, 10, 0.95)",
                borderColor: "#333333",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#fff",
                fontFamily: "monospace",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "6px", fontFamily: "monospace" }} />
            {showActual && (
              <Area
                type="monotone"
                dataKey="actual"
                name="Historical Baseline"
                stroke="#52525B"
                fillOpacity={0.1}
                fill="#52525B"
                strokeWidth={1.5}
                strokeDasharray="2 2"
              />
            )}
            {showGbdt && (
              <Area
                type="monotone"
                dataKey="predicted"
                name="GBDT Trio Forecast"
                stroke="#FFFFFF"
                fillOpacity={1}
                fill="url(#gbdtGrad)"
                strokeWidth={2.5}
              />
            )}
            {showGnn && (
              <Area
                type="monotone"
                dataKey="st_gnn"
                name="ST-GNN Neural Forecast"
                stroke="#A1A1AA"
                fillOpacity={1}
                fill="url(#gnnGrad)"
                strokeWidth={1.5}
                strokeDasharray="4 4"
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="flex items-center justify-between text-[10px] text-zinc-400 font-mono pt-2 border-t border-zinc-800">
        <span>Target Horizon: t+{horizon} ({horizon * 15} min)</span>
        <span className="text-white font-bold">Predicted Demand: {currentPrediction} Rides</span>
      </div>
    </div>
  );
}
