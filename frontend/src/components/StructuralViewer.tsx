import React, { useEffect, useRef } from 'react';
import {
  Atom,
  CheckCircle,
  Loader2,
  Maximize2,
  Minimize2,
  Pause,
  Play,
} from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { PluginCommands } from 'molstar/lib/mol-plugin/commands';
import 'molstar/lib/mol-plugin-ui/skin/light.scss';
import { AffinityState, highlightColor } from '../utils/bindingAnimation';

type Props = {
  pdbUrl?: string;
  affinityState: AffinityState;
  measuredKd?: number;
  isPlaying: boolean;
  onToggleAnimation: () => void;
  isLoading?: boolean;
};

const affinityLabelMap: Record<AffinityState, string> = {
  strong: 'High Affinity',
  moderate: 'Moderate Affinity',
  weak: 'Low Affinity',
  unknown: 'Affinity Pending',
};

export default function StructuralViewer({
  pdbUrl,
  affinityState,
  measuredKd,
  isPlaying,
  onToggleAnimation,
  isLoading,
}: Props) {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pluginRef = useRef<any>(null);
  const pluginInitRef = useRef<Promise<any> | null>(null);

  const ensurePlugin = async () => {
    if (pluginRef.current) {
      return pluginRef.current;
    }
    if (!viewerRef.current) {
      return null;
    }
    if (!pluginInitRef.current) {
      pluginInitRef.current = createPluginUI({
        target: viewerRef.current,
        spec: DefaultPluginUISpec(),
        render: renderReact18,
      }).then((plugin) => {
        pluginRef.current = plugin;
        return plugin;
      });
    }
    return pluginInitRef.current;
  };

  const handleReset = async () => {
    const plugin = await ensurePlugin();
    if (!plugin) {
      return;
    }
    PluginCommands.Camera.Reset(plugin, { durationMs: 250 });
  };

  const handleFullscreen = () => {
    if (!containerRef.current) {
      return;
    }
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen?.();
      return;
    }
    document.exitFullscreen?.();
  };

  useEffect(() => {
    void ensurePlugin();
    return () => {
      if (pluginRef.current?.dispose) {
        pluginRef.current.dispose();
      }
      pluginRef.current = null;
      pluginInitRef.current = null;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loadStructure = async () => {
      const plugin = await ensurePlugin();
      if (!plugin) {
        return;
      }
      plugin.clear();
      if (!pdbUrl) {
        return;
      }
      try {
        const data = await plugin.builders.data.download(
          { url: pdbUrl },
          { state: { isGhost: true } }
        );
        const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
        await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
        if (cancelled) {
          return;
        }
        plugin.canvas3d?.setProps({
          highlightColor: highlightColor(affinityState),
          backgroundColor: 'white',
        });
      } catch (error) {
        console.error('Failed to load structure', pdbUrl, error);
      }
    };
    void loadStructure();
    return () => {
      cancelled = true;
    };
  }, [pdbUrl, affinityState]);

  const affinityLabel = affinityLabelMap[affinityState];
  const kdValue = typeof measuredKd === 'number' ? `${measuredKd.toFixed(1)} nM` : 'n/a';

  const showLoading = Boolean(isLoading);
  const showEmpty = !pdbUrl && !showLoading;

  return (
    <div
      className="viewer-container"
      ref={containerRef}
      aria-live="polite"
      aria-busy={isLoading}
    >
      <div className="viewer-header">
        <div className="viewer-title">
          <Atom size={24} color="#4286F5" aria-hidden="true" />
          <span>Fc-FcγR Complex</span>
        </div>
        <div className="viewer-controls">
          <button
            className="icon-btn"
            type="button"
            title={isPlaying ? 'Pause animation' : 'Play animation'}
            aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
            onClick={onToggleAnimation}
            disabled={!pdbUrl}
          >
            {isPlaying ? <Pause size={20} aria-hidden="true" /> : <Play size={20} aria-hidden="true" />}
          </button>
          <button className="icon-btn" type="button" title="Reset view" aria-label="Reset view" onClick={handleReset}>
            <Minimize2 size={20} aria-hidden="true" />
          </button>
          <button
            className="icon-btn"
            type="button"
            title="Toggle fullscreen"
            aria-label="Toggle fullscreen"
            onClick={handleFullscreen}
          >
            <Maximize2 size={20} aria-hidden="true" />
          </button>
        </div>
      </div>
      <div className="viewer-body">
        <div ref={viewerRef} className="viewer-canvas" aria-label="Molecular structure viewer" />
        {showEmpty && (
          <div className="viewer-placeholder">
            Select an FcγR allotype and glycan variant, then Predict Binding.
          </div>
        )}
        {showLoading && (
          <div className="viewer-placeholder">
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
              <Loader2 size={18} className="spinner" aria-hidden="true" />
              Loading structure...
            </div>
          </div>
        )}
      </div>
      <div className="viewer-status">
        <span className={`status-chip ${affinityState}`}>
          <CheckCircle size={14} aria-hidden="true" />
          {affinityLabel} (K
          <sub>D</sub> = {kdValue})
        </span>
      </div>
    </div>
  );
}
