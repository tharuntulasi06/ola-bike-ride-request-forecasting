"use client";

import React, { useEffect, useState } from "react";
import { ClusterInfo } from "../types";
import { MapPin, Navigation, Layers, Search } from "lucide-react";

interface SpatialMapProps {
  clusters: ClusterInfo[];
  selectedClusterId: number;
  onSelectCluster: (clusterId: number) => void;
}

export default function SpatialMap({
  clusters,
  selectedClusterId,
  onSelectCluster,
}: SpatialMapProps) {
  const [isClient, setIsClient] = useState(false);
  const [MapComponents, setMapComponents] = useState<any>(null);
  const [mapMode, setMapMode] = useState<"centroids" | "heat" | "flow">("centroids");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    setIsClient(true);
    Promise.all([
      import("react-leaflet"),
      import("leaflet")
    ]).then(([reactLeaflet, L]) => {
      setMapComponents({
        MapContainer: reactLeaflet.MapContainer,
        TileLayer: reactLeaflet.TileLayer,
        CircleMarker: reactLeaflet.CircleMarker,
        Polyline: reactLeaflet.Polyline,
        Popup: reactLeaflet.Popup,
        Tooltip: reactLeaflet.Tooltip,
        L,
      });
    });
  }, []);

  if (!isClient || !MapComponents) {
    return (
      <div className="liquid-glass p-6 h-[440px] flex flex-col items-center justify-center text-zinc-400">
        <Navigation className="w-8 h-8 animate-spin text-white mb-2" />
        <p className="text-xs font-mono">Loading Chennai Grayscale Spatial Map Engine...</p>
      </div>
    );
  }

  const { MapContainer, TileLayer, CircleMarker, Polyline, Popup, Tooltip } = MapComponents;
  const chennaiCenter: [number, number] = [13.0418, 80.2341];

  const filteredClusters = clusters.filter((c) =>
    c.landmark_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Vector flow lines for "flow" mode
  const flowLines = [
    { from: [13.0067, 80.2020], to: [13.0418, 80.2341] }, // Guindy -> T. Nagar
    { from: [13.0827, 80.2707], to: [12.9645, 80.2443] }, // Central -> OMR
    { from: [12.9750, 80.2207], to: [13.0850, 80.2101] }, // Velachery -> CMBT
  ];

  return (
    <div className="liquid-glass p-4 h-[440px] flex flex-col justify-between">
      {/* Map Control Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <MapPin className="w-4 h-4 text-white" />
          <h2 className="text-xs font-bold text-white uppercase tracking-wider font-mono">
            Chennai Spatial Centroid Map
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {/* Interactive Search Box */}
          <div className="relative">
            <Search className="w-3 h-3 absolute left-2 top-2 text-zinc-400" />
            <input
              type="text"
              placeholder="Filter landmark..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-7 pr-2 py-1 text-[11px] bg-zinc-900 border border-zinc-800 rounded-md text-white placeholder-zinc-500 font-mono focus:outline-none focus:border-white"
            />
          </div>

          {/* Interactive Map Layer Switcher */}
          <div className="flex items-center bg-zinc-900 p-0.5 rounded-lg border border-zinc-800 text-[11px] font-mono">
            <button
              onClick={() => setMapMode("centroids")}
              className={`px-2 py-1 rounded ${
                mapMode === "centroids" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              Grid
            </button>
            <button
              onClick={() => setMapMode("heat")}
              className={`px-2 py-1 rounded ${
                mapMode === "heat" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              Surge
            </button>
            <button
              onClick={() => setMapMode("flow")}
              className={`px-2 py-1 rounded ${
                mapMode === "flow" ? "bg-white text-black font-bold" : "text-zinc-400 hover:text-white"
              }`}
            >
              Flows
            </button>
          </div>
        </div>
      </div>

      {/* Map Container */}
      <div className="flex-1 rounded-xl overflow-hidden relative border border-zinc-800">
        <MapContainer
          center={chennaiCenter}
          zoom={11}
          scrollWheelZoom={true}
          style={{ height: "100%", width: "100%" }}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.esri.com/">Esri</a>, HERE, Garmin, USGS'
            url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          />

          {/* Flow Lines */}
          {mapMode === "flow" &&
            flowLines.map((line, i) => (
              <Polyline
                key={i}
                positions={[line.from as [number, number], line.to as [number, number]]}
                pathOptions={{ color: "#FFFFFF", weight: 2, dashArray: "6 6" }}
              />
            ))}

          {/* Cluster Markers */}
          {filteredClusters.map((c) => {
            const isSelected = c.cluster_id === selectedClusterId;
            const radius = mapMode === "heat" ? 28 : isSelected ? 22 : 14;

            return (
              <CircleMarker
                key={c.cluster_id}
                center={[c.latitude, c.longitude]}
                radius={radius}
                pathOptions={{
                  fillColor: isSelected ? "#FFFFFF" : "#52525B",
                  fillOpacity: isSelected ? 0.9 : 0.6,
                  color: "#FFFFFF",
                  weight: isSelected ? 3 : 1,
                }}
                eventHandlers={{
                  click: () => onSelectCluster(c.cluster_id),
                }}
              >
                <Tooltip permanent direction="top" offset={[0, -10]}>
                  <span className="text-[10px] font-bold text-white font-mono bg-black/80 px-1.5 py-0.5 rounded border border-zinc-700">
                    {c.landmark_name.replace(/_/g, " ").toUpperCase()}
                  </span>
                </Tooltip>
                <Popup>
                  <div className="p-1 font-mono">
                    <h4 className="font-bold text-xs text-white">
                      Cluster #{c.cluster_id}: {c.landmark_name}
                    </h4>
                    <p className="text-[10px] text-zinc-400 mt-1">
                      Lat: {c.latitude.toFixed(4)} | Lon: {c.longitude.toFixed(4)}
                    </p>
                    <button
                      onClick={() => onSelectCluster(c.cluster_id)}
                      className="mt-2 w-full py-1 text-[11px] rounded bg-white text-black font-bold hover:bg-zinc-200 transition-colors"
                    >
                      Set Active Focus
                    </button>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>

      <div className="pt-2 flex justify-between items-center text-[10px] text-zinc-400 font-mono">
        <span>Active Map Overlay: {mapMode.toUpperCase()} VIEW</span>
        <span className="text-white">Selected Zone #{selectedClusterId}</span>
      </div>
    </div>
  );
}
