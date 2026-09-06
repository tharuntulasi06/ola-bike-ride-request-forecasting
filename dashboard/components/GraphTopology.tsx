"use client";

import React, { useState } from "react";
import { Share2, Sliders } from "lucide-react";

interface GraphTopologyProps {
  selectedClusterId: number;
  onSelectCluster: (id: number) => void;
}

export default function GraphTopology({ selectedClusterId, onSelectCluster }: GraphTopologyProps) {
  const [minWeight, setMinWeight] = useState<number>(0.75);

  const nodes = [
    { id: 0, label: "Central", x: 50, y: 30 },
    { id: 1, label: "T. Nagar", x: 120, y: 70 },
    { id: 2, label: "OMR IT", x: 190, y: 130 },
    { id: 3, label: "Velachery", x: 140, y: 140 },
    { id: 4, label: "Guindy", x: 70, y: 110 },
    { id: 5, label: "CMBT", x: 30, y: 70 },
  ];

  const edges = [
    { source: 0, target: 1, weight: 0.82 },
    { source: 1, target: 4, weight: 0.91 },
    { source: 4, target: 3, weight: 0.85 },
    { source: 3, target: 2, weight: 0.95 },
    { source: 0, target: 5, weight: 0.78 },
    { source: 1, target: 3, weight: 0.88 },
  ];

  const filteredEdges = edges.filter((e) => e.weight >= minWeight);

  return (
    <div className="liquid-glass p-4 h-[340px] flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono flex items-center gap-2">
            <Share2 className="w-4 h-4 text-white" />
            ST-GNN Graph Topology
          </h2>
          <span className="text-[10px] font-mono text-zinc-400">Matrix W_ij</span>
        </div>
        <p className="text-[11px] text-zinc-400 font-mono mb-2">
          Click graph nodes to focus active cluster matrix state
        </p>

        {/* Interactive Threshold Slider */}
        <div className="flex items-center justify-between gap-2 text-[10px] font-mono bg-zinc-900 p-1.5 rounded-md border border-zinc-800">
          <span className="text-zinc-400 flex items-center gap-1">
            <Sliders className="w-3 h-3 text-white" /> Edge Weight &gt;=
          </span>
          <input
            type="range"
            min="0.70"
            max="0.95"
            step="0.05"
            value={minWeight}
            onChange={(e) => setMinWeight(parseFloat(e.target.value))}
            className="w-20 h-1 bg-zinc-800 rounded appearance-none accent-white cursor-pointer"
          />
          <span className="text-white font-bold">{minWeight.toFixed(2)}</span>
        </div>
      </div>

      <div className="flex-1 my-2 bg-zinc-950/80 rounded-xl border border-zinc-800 relative flex items-center justify-center overflow-hidden">
        <svg className="w-full h-full" viewBox="0 0 240 170">
          {/* Edges */}
          {filteredEdges.map((e, idx) => {
            const s = nodes.find((n) => n.id === e.source)!;
            const t = nodes.find((n) => n.id === e.target)!;
            const isConnectedToSelected = e.source === selectedClusterId || e.target === selectedClusterId;
            return (
              <line
                key={idx}
                x1={s.x}
                y1={s.y}
                x2={t.x}
                y2={t.y}
                stroke={isConnectedToSelected ? "#FFFFFF" : "#404040"}
                strokeWidth={isConnectedToSelected ? "2.5" : "1"}
                strokeDasharray={isConnectedToSelected ? "none" : "2 2"}
              />
            );
          })}

          {/* Interactive Nodes */}
          {nodes.map((n) => {
            const isSelected = n.id === selectedClusterId;
            return (
              <g key={n.id} onClick={() => onSelectCluster(n.id)} className="cursor-pointer group">
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={isSelected ? "13" : "8"}
                  fill={isSelected ? "#FFFFFF" : "#262626"}
                  stroke="#FFFFFF"
                  strokeWidth={isSelected ? "2" : "1"}
                  className="transition-all duration-200 group-hover:scale-125"
                />
                <text
                  x={n.x}
                  y={n.y + (isSelected ? 20 : 16)}
                  textAnchor="middle"
                  fill={isSelected ? "#FFFFFF" : "#A1A1AA"}
                  fontSize={isSelected ? "10" : "8"}
                  fontFamily="monospace"
                  fontWeight={isSelected ? "bold" : "normal"}
                >
                  {n.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="pt-2 border-t border-zinc-800 flex justify-between items-center text-[10px] text-zinc-400 font-mono">
        <span>Active Node: #{selectedClusterId}</span>
        <span className="text-white font-bold">{filteredEdges.length} Active Spatial Edges</span>
      </div>
    </div>
  );
}
