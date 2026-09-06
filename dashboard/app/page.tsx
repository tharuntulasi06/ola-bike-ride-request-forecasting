"use client";

import React, { useEffect, useState, useCallback } from "react";
import Header from "../components/Header";
import KPICards from "../components/KPICards";
import SpatialMap from "../components/SpatialMap";
import DemandChart from "../components/DemandChart";
import WhatIfSimulator from "../components/WhatIfSimulator";
import RebalanceTable from "../components/RebalanceTable";
import SqlExplorer from "../components/SqlExplorer";
import GraphTopology from "../components/GraphTopology";
import ShapPanel from "../components/ShapPanel";

import {
  fetchHealth,
  fetchClusters,
  fetchPrediction,
  fetchRebalanceAdvice,
  fetchDbSummary,
  fetchMetrics,
  fetchShap,
} from "../lib/api";

import {
  ClusterInfo,
  RebalanceRecommendation,
  DbZoneSummary,
  EvaluationMetrics,
  FeatureImportance,
} from "../types";

export default function DashboardPage() {
  const [apiStatus, setApiStatus] = useState<string>("connecting...");
  const [activeView, setActiveView] = useState<"operational" | "analytics">("operational");
  const [activeKpiFilter, setActiveKpiFilter] = useState<string | null>(null);

  const [clusters, setClusters] = useState<ClusterInfo[]>([]);
  const [selectedClusterId, setSelectedClusterId] = useState<number>(0);
  const [currentPrediction, setCurrentPrediction] = useState<number>(45);
  const [actualSurge, setActualSurge] = useState<number>(1.15);
  const [predictedSurge, setPredictedSurge] = useState<number>(1.45);
  const [horizon, setHorizon] = useState<number>(1);
  const [temp, setTemp] = useState<number>(30.5);
  const [rain, setRain] = useState<number>(0.0);
  const [rebalanceRecs, setRebalanceRecs] = useState<RebalanceRecommendation[]>([]);
  const [dbSummary, setDbSummary] = useState<DbZoneSummary[]>([]);
  const [metrics, setMetrics] = useState<EvaluationMetrics | null>(null);
  const [shapFeatures, setShapFeatures] = useState<FeatureImportance[]>([]);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  const runPrediction = useCallback(async () => {
    setIsSimulating(true);
    const pred = await fetchPrediction(selectedClusterId, horizon, temp, rain);
    setCurrentPrediction(pred.predicted_demand);
    if (pred.actual_surge) setActualSurge(pred.actual_surge);
    if (pred.predicted_surge) setPredictedSurge(pred.predicted_surge);
    setIsSimulating(false);
  }, [selectedClusterId, horizon, temp, rain]);

  const loadAllData = useCallback(async () => {
    setIsRefreshing(true);
    const health = await fetchHealth();
    setApiStatus(health.status);

    const clusterData = await fetchClusters();
    setClusters(clusterData.clusters);

    const recs = await fetchRebalanceAdvice();
    setRebalanceRecs(recs);

    const dbData = await fetchDbSummary();
    setDbSummary(dbData);

    const met = await fetchMetrics();
    setMetrics(met);

    const shap = await fetchShap();
    setShapFeatures(shap);

    await runPrediction();

    setTimeout(() => {
      setIsRefreshing(false);
    }, 600);
  }, [runPrediction]);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  useEffect(() => {
    runPrediction();
  }, [runPrediction]);

  const activeCluster = clusters.find((c) => c.cluster_id === selectedClusterId);
  const activeClusterName = activeCluster ? activeCluster.landmark_name : "chennai_central";

  const handleResetSimulator = () => {
    setTemp(30.5);
    setRain(0.0);
  };

  return (
    <div className="min-h-screen bg-monoBlack text-white p-4 md:p-6 max-w-[1600px] mx-auto space-y-6">
      {/* Header & Mode Switcher */}
      <Header
        apiStatus={apiStatus}
        activeView={activeView}
        onViewChange={(view) => setActiveView(view)}
        onRefresh={loadAllData}
        isRefreshing={isRefreshing}
      />

      {/* Telemetry KPI Cards */}
      <KPICards
        metrics={metrics}
        activeClusterName={activeClusterName}
        activeKpiFilter={activeKpiFilter}
        onSelectKpiFilter={(f) => setActiveKpiFilter(f)}
        actualSurge={actualSurge}
        predictedSurge={predictedSurge}
      />

      {/* View Mode 1: Operational Control View */}
      {activeView === "operational" && (
        <>
          {/* Row 1: Spatial Map & Multi-Horizon Forecast */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-6">
              <SpatialMap
                clusters={clusters}
                selectedClusterId={selectedClusterId}
                onSelectCluster={(id) => setSelectedClusterId(id)}
              />
            </div>
            <div className="lg:col-span-6">
              <DemandChart
                clusterName={activeClusterName}
                clusterId={selectedClusterId}
                currentPrediction={currentPrediction}
                horizon={horizon}
                onHorizonChange={(h) => setHorizon(h)}
              />
            </div>
          </div>

          {/* Row 2: What-If Simulator & Rebalancing Engine */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-5">
              <WhatIfSimulator
                temp={temp}
                rain={rain}
                onTempChange={(t) => setTemp(t)}
                onRainChange={(r) => setRain(r)}
                onReset={handleResetSimulator}
                isSimulating={isSimulating}
              />
            </div>
            <div className="lg:col-span-7">
              <RebalanceTable recommendations={rebalanceRecs} />
            </div>
          </div>

          {/* Row 3: DuckDB SQL, ST-GNN Topology & SHAP Explainability */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-6">
              <SqlExplorer data={dbSummary} />
            </div>
            <div className="lg:col-span-3">
              <GraphTopology
                selectedClusterId={selectedClusterId}
                onSelectCluster={(id) => setSelectedClusterId(id)}
              />
            </div>
            <div className="lg:col-span-3">
              <ShapPanel features={shapFeatures} />
            </div>
          </div>
        </>
      )}

      {/* View Mode 2: Full Analytics Workbench View */}
      {activeView === "analytics" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-8">
            <SqlExplorer data={dbSummary} />
          </div>
          <div className="lg:col-span-4 space-y-6">
            <GraphTopology
              selectedClusterId={selectedClusterId}
              onSelectCluster={(id) => setSelectedClusterId(id)}
            />
            <ShapPanel features={shapFeatures} />
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="pt-6 border-t border-zinc-900 text-center text-xs text-zinc-500 font-mono">
        OLA RIDE FORECASTING SYSTEM &bull; MONOCHROME LIQUID GLASS CONTROL CENTER
      </footer>
    </div>
  );
}
