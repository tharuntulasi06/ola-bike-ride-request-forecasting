"use client";

import React, { useState } from "react";
import { Brain, Info } from "lucide-react";
import { FeatureImportance } from "../types";

interface ShapPanelProps {
  features: FeatureImportance[];
}

export default function ShapPanel({ features }: ShapPanelProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>("All");
  const [activeFeatureModal, setActiveFeatureModal] = useState<string | null>(null);

  const categories = ["All", "Temporal", "Historical", "Weather", "Spatial"];

  const filteredFeatures = features.filter((f) => {
    if (selectedCategory === "All") return true;
    return f.category === selectedCategory;
  });

  return (
    <div className="liquid-glass p-4 h-[340px] flex flex-col justify-between relative">
      <div>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
            <Brain className="w-4 h-4 text-white" />
            SHAP Explainability
          </h2>
          <span className="text-[10px] font-mono text-zinc-400">Tweedie Loss</span>
        </div>
        <p className="text-[11px] text-zinc-400 font-mono mb-2">
          Shapley feature impacts on ride request forecasts
        </p>

        {/* Interactive Category Filter Tabs */}
        <div className="flex items-center gap-1 bg-zinc-900 p-0.5 rounded-lg border border-zinc-800 text-[10px] font-mono mb-2 overflow-x-auto">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-2 py-0.5 rounded transition-all ${
                selectedCategory === cat ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </div>

      {/* Feature Progress Bars */}
      <div className="space-y-2.5 my-1 overflow-y-auto pr-1 flex-1">
        {filteredFeatures.map((f, idx) => {
          const pct = Math.round(f.importance * 100);
          return (
            <div
              key={idx}
              onClick={() => setActiveFeatureModal(activeFeatureModal === f.feature ? null : f.feature)}
              className="space-y-1 cursor-pointer group"
            >
              <div className="flex justify-between text-xs font-mono">
                <span className="text-zinc-300 group-hover:text-white transition-colors flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-white"></span>
                  {f.feature}
                </span>
                <span className="text-white font-bold">{pct}%</span>
              </div>
              <div className="w-full h-1.5 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
                <div
                  className="h-full bg-white rounded-full transition-all duration-500"
                  style={{ width: `${pct}%` }}
                ></div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="pt-2 border-t border-zinc-800 flex justify-between items-center text-[10px] text-zinc-400 font-mono">
        <span className="flex items-center gap-1">
          <Info className="w-3 h-3 text-white" /> Driver:
        </span>
        <span className="text-white font-bold">Hour of Day (32%)</span>
      </div>
    </div>
  );
}
