import React, { useEffect, useRef } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { PredictionRecord } from '../types';
import { structureUrl } from '../api';
import { classifyAffinity, highlightColor } from '../utils/bindingAnimation';

interface Props {
  fcgrName: string;
  data: PredictionRecord[];
}

export default function ComparisonGrid({ fcgrName, data }: Props) {
  const viewerRefs = useRef<Array<HTMLDivElement | null>>([]);
  const pluginRefs = useRef<Array<any>>([]);

  useEffect(() => {
    let cancelled = false;

    const ensurePlugin = async (index: number, container: HTMLDivElement) => {
      if (pluginRefs.current[index]) {
        return pluginRefs.current[index];
      }
      const plugin = await createPluginUI(container, DefaultPluginUISpec());
      pluginRefs.current[index] = plugin;
      return plugin;
    };

    const loadAll = async () => {
      await Promise.all(
        data.map(async (row, index) => {
          const container = viewerRefs.current[index];
          if (!container) {
            return;
          }
          const plugin = await ensurePlugin(index, container);
          if (!plugin || cancelled) {
            return;
          }
          const pdbUrl = structureUrl(fcgrName, row.glycan_name, 'pdb');
          try {
            plugin.clear();
            const dataRef = await plugin.builders.data.download(
              { url: pdbUrl },
              { state: { isGhost: true } }
            );
            const trajectory = await plugin.builders.structure.parseTrajectory(dataRef, 'pdb');
            await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
            if (cancelled) {
              return;
            }
            const affinity = classifyAffinity(row.binding_kd_nm);
            plugin.canvas3d?.setProps({
              highlightColor: highlightColor(affinity),
              backgroundColor: 'white',
            });
          } catch (error) {
            console.error('Failed to load structure', pdbUrl, error);
          }
        })
      );
    };

    void loadAll();

    return () => {
      cancelled = true;
      pluginRefs.current.forEach((plugin) => plugin?.dispose?.());
      pluginRefs.current = [];
    };
  }, [data, fcgrName]);

  return (
    <div className="comparison-grid">
      {data.map((row, index) => {
        const affinity = classifyAffinity(row.binding_kd_nm);
        return (
          <div key={`${row.fcgr_name}-${row.glycan_name}`} className="comparison-card">
            <div className="comparison-header">
              <span>{row.glycan_name}</span>
              <span className={`badge badge-${affinity}`}>{affinity}</span>
            </div>
            <div
              ref={(el) => {
                viewerRefs.current[index] = el;
              }}
              className="comparison-viewer"
              style={{ borderColor: highlightColor(affinity) }}
            />
          </div>
        );
      })}
    </div>
  );
}
