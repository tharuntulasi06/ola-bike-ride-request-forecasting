"use client";

import React from "react";
import { CloudRain, Thermometer, RefreshCw, Zap } from "lucide-react";

interface WhatIfSimulatorProps {
  temp: number;
  rain: number;
  onTempChange: (t: number) => void;
  onRainChange: (r: number) => void;
  onReset: () => void;
  isSimulating: boolean;
}

export default function WhatIfSimulator({
  temp,
  rain,
  onTempChange,
  onRainChange,
  onReset,
  isSimulating,
}: WhatIfSimulatorProps) {
  const applyPreset = (presetTemp: number, presetRain: number) => {
    onTempChange(presetTemp);
    onRainChange(presetRain);
  };

  return (
    <div className="liquid-glass p-4 h-[340px] flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
            <CloudRain className="w-4 h-4 text-white" />
            "What-If" Weather Simulator
          </h2>
          <button
            onClick={onReset}
            className="flex items-center gap-1 text-[10px] text-zinc-400 hover:text-white transition-colors font-mono"
          >
            <RefreshCw className="w-3 h-3" /> Reset Defaults
          </button>
        </div>
        <p className="text-[11px] text-zinc-400 font-mono mb-3">
          Simulate weather impact parameters on spatiotemporal ride demand models
        </p>

        {/* Interactive Scenario Presets */}
        <div className="flex items-center gap-1.5 flex-wrap mb-4">
          <button
            onClick={() => applyPreset(24, 40)}
            className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-[10px] font-mono hover:text-white transition-all"
          >
            🌧️ Downpour
          </button>
          <button
            onClick={() => applyPreset(42, 0)}
            className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-[10px] font-mono hover:text-white transition-all"
          >
            🔥 Heatwave
          </button>
          <button
            onClick={() => applyPreset(31, 8)}
            className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-[10px] font-mono hover:text-white transition-all"
          >
            ⚡ Rush Surge
          </button>
          <button
            onClick={() => applyPreset(28, 0)}
            className="px-2 py-1 rounded bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-zinc-300 text-[10px] font-mono hover:text-white transition-all"
          >
            ☀️ Clear
          </button>
        </div>
      </div>

      {/* Sliders */}
      <div className="space-y-4">
        {/* Temperature */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-zinc-400 flex items-center gap-1">
              <Thermometer className="w-3.5 h-3.5 text-white" /> Temperature
            </span>
            <span className="text-white font-bold">{temp}°C</span>
          </div>
          <input
            type="range"
            min="15"
            max="45"
            step="0.5"
            value={temp}
            onChange={(e) => onTempChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-zinc-900 rounded-lg appearance-none cursor-pointer accent-white"
          />
          <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
            <span>15°C</span>
            <span>30°C</span>
            <span>45°C</span>
          </div>
        </div>

        {/* Rain */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-zinc-400 flex items-center gap-1">
              <CloudRain className="w-3.5 h-3.5 text-white" /> Rainfall Intensity
            </span>
            <span className="text-white font-bold">{rain} mm/h</span>
          </div>
          <input
            type="range"
            min="0"
            max="50"
            step="1"
            value={rain}
            onChange={(e) => onRainChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-zinc-900 rounded-lg appearance-none cursor-pointer accent-white"
          />
          <div className="flex justify-between text-[9px] text-zinc-500 font-mono">
            <span>0 mm/h</span>
            <span>25 mm/h</span>
            <span>50 mm/h</span>
          </div>
        </div>
      </div>

      <div className="pt-2 border-t border-zinc-800 flex justify-between items-center text-[10px] font-mono">
        <span className="text-zinc-400">Simulation Status:</span>
        <span className={isSimulating ? "text-white font-bold animate-pulse" : "text-zinc-300"}>
          {isSimulating ? "Recalculating REST Model..." : "Live Inference Synced"}
        </span>
      </div>
    </div>
  );
}
