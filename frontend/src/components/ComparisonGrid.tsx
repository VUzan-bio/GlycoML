import React, { useEffect, useRef } from 'react';
import { Atom } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import 'molstar/lib/mol-plugin-ui/skin/light.scss';
import { PredictionRecord } from '../types';
import { structureUrl } from '../api';
import { classifyAffinity, highlightColor } from '../utils/bindingAnimation';
import Card from './ui/Card';
import styles from './ComparisonGrid.module.css';

interface Props {
  fcgrName: string;
  data: PredictionRecord[];
}

export default function ComparisonGrid({ fcgrName, data }: Props) {
  const viewerRefs = useRef<Array<HTMLDivElement | null>>([]);
  const pluginRefs = useRef<Array<any>>([]);
  const pluginInitRefs = useRef<Array<Promise<any> | null>>([]);

  useEffect(() => {
    let cancelled = false;

    const ensurePlugin = async (index: number, container: HTMLDivElement) => {
      if (pluginRefs.current[index]) {
        return pluginRefs.current[index];
      }
      if (!pluginInitRefs.current[index]) {
        pluginInitRefs.current[index] = createPluginUI({
          target: container,
          spec: DefaultPluginUISpec(),
          render: renderReact18,
        }).then((plugin) => {
          pluginRefs.current[index] = plugin;
          return plugin;
        });
      }
      return pluginInitRefs.current[index];
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
      pluginInitRefs.current = [];
    };
  }, [data, fcgrName]);

  return (
    <div className="comparison-grid">
      {data.map((row, index) => {
        const affinity = classifyAffinity(row.binding_kd_nm);
        return (
          <Card key={`${row.fcgr_name}-${row.glycan_name}`} className={styles.card}>
            <div className={styles.header}>
              <span className={styles.title}>
                <Atom size={14} color="#4286F5" aria-hidden="true" /> {row.glycan_name}
              </span>
              <span className={`badge badge-${affinity}`}>{affinity}</span>
            </div>
            <div
              ref={(el) => {
                viewerRefs.current[index] = el;
              }}
              className={styles.viewer}
              style={{ borderColor: highlightColor(affinity) }}
              aria-label={`Structure viewer for ${row.glycan_name}`}
            />
            {row.structure?.has_glycan === false && (
              <span className={styles.templateBadge}>Template only</span>
            )}
          </Card>
        );
      })}
    </div>
  );
}
