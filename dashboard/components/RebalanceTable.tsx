"use client";

import React, { useState } from "react";
import { Truck, ArrowRight, CheckCircle2, ArrowUpDown } from "lucide-react";
import { RebalanceRecommendation } from "../types";

interface RebalanceTableProps {
  recommendations: RebalanceRecommendation[];
}

export default function RebalanceTable({ recommendations }: RebalanceTableProps) {
  const [dispatchedIds, setDispatchedIds] = useState<number[]>([]);
  const [priorityFilter, setPriorityFilter] = useState<"All" | "High" | "Medium">("All");
  const [sortField, setSortField] = useState<"qty" | "time" | "revenue">("revenue");

  const handleDispatch = (idx: number) => {
    setDispatchedIds((prev) => [...prev, idx]);
  };

  const filteredRecs = recommendations.filter((r) => {
    if (priorityFilter === "All") return true;
    return r.priority === priorityFilter;
  });

  const sortedRecs = [...filteredRecs].sort((a, b) => {
    if (sortField === "qty") return b.recommended_transfer_qty - a.recommended_transfer_qty;
    if (sortField === "time") return a.estimated_transit_time_mins - b.estimated_transit_time_mins;
    return b.revenue_uplift_inr - a.revenue_uplift_inr;
  });

  return (
    <div className="liquid-glass p-4 h-[340px] flex flex-col justify-between">
      <div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
          <div className="flex items-center gap-2">
            <Truck className="w-4 h-4 text-white" />
            <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
              Automated Fleet Rebalancing Matrix
            </h2>
          </div>

          {/* Interactive Priority Filter Tabs */}
          <div className="flex items-center bg-zinc-900 p-0.5 rounded-lg border border-zinc-800 text-[10px] font-mono">
            <button
              onClick={() => setPriorityFilter("All")}
              className={`px-2 py-0.5 rounded ${
                priorityFilter === "All" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              All
            </button>
            <button
              onClick={() => setPriorityFilter("High")}
              className={`px-2 py-0.5 rounded ${
                priorityFilter === "High" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              High Priority
            </button>
            <button
              onClick={() => setPriorityFilter("Medium")}
              className={`px-2 py-0.5 rounded ${
                priorityFilter === "Medium" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              Medium
            </button>
          </div>
        </div>
        <p className="text-[11px] text-zinc-400 font-mono mb-2">
          Spatial route vectors to rebalance fleet surplus to deficit demand zones
        </p>
      </div>

      <div className="overflow-x-auto flex-1">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-zinc-800 text-[10px] text-zinc-400 uppercase font-mono">
              <th className="py-1.5 px-2">Origin Surplus</th>
              <th className="py-1.5 px-2">Destination Deficit</th>
              <th
                onClick={() => setSortField("qty")}
                className="py-1.5 px-2 text-center cursor-pointer hover:text-white transition-colors"
              >
                <span className="flex items-center justify-center gap-1">
                  Bikes <ArrowUpDown className="w-3 h-3" />
                </span>
              </th>
              <th
                onClick={() => setSortField("time")}
                className="py-1.5 px-2 text-center cursor-pointer hover:text-white transition-colors"
              >
                <span className="flex items-center justify-center gap-1">
                  Time <ArrowUpDown className="w-3 h-3" />
                </span>
              </th>
              <th
                onClick={() => setSortField("revenue")}
                className="py-1.5 px-2 text-right cursor-pointer hover:text-white transition-colors"
              >
                <span className="flex items-center justify-end gap-1">
                  Uplift <ArrowUpDown className="w-3 h-3" />
                </span>
              </th>
              <th className="py-1.5 px-2 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/60 text-xs font-mono">
            {sortedRecs.map((rec, idx) => {
              const isDispatched = dispatchedIds.includes(idx);
              return (
                <tr key={idx} className="hover:bg-zinc-900/60 transition-colors">
                  <td className="py-2 px-2 text-zinc-300">
                    {rec.origin_name.replace(/_/g, " ")}
                  </td>
                  <td className="py-2 px-2 text-white font-medium">
                    <span className="flex items-center gap-1">
                      <ArrowRight className="w-3 h-3 text-zinc-500" />
                      {rec.destination_name.replace(/_/g, " ")}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-center font-bold text-white">
                    +{rec.recommended_transfer_qty}
                  </td>
                  <td className="py-2 px-2 text-center text-zinc-400">
                    {rec.estimated_transit_time_mins}m
                  </td>
                  <td className="py-2 px-2 text-right text-white font-bold">
                    ₹{rec.revenue_uplift_inr}
                  </td>
                  <td className="py-2 px-2 text-center">
                    {isDispatched ? (
                      <span className="inline-flex items-center gap-1 text-[10px] text-white font-bold bg-zinc-800 px-2 py-0.5 rounded border border-zinc-700">
                        <CheckCircle2 className="w-3 h-3 text-white" /> En Route
                      </span>
                    ) : (
                      <button
                        onClick={() => handleDispatch(idx)}
                        className="liquid-btn px-2.5 py-1 text-[10px] rounded font-bold transition-all active:scale-95"
                      >
                        Dispatch
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="pt-2 border-t border-zinc-800 flex justify-between items-center text-[10px] text-zinc-400 font-mono">
        <span>Sort By: {sortField.toUpperCase()}</span>
        <span className="text-white font-bold">+₹3,040 Revenue Uplift</span>
      </div>
    </div>
  );
}
