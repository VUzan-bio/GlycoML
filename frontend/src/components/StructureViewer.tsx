import React, { useEffect, useRef } from 'react';
import { Eye } from 'lucide-react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { renderReact18 } from 'molstar/lib/mol-plugin-ui/react18';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import 'molstar/lib/mol-plugin-ui/skin/light.scss';

import { highlightColor } from '../utils/bindingAnimation';
import Card from './ui/Card';

type Props = {
  pdbUrl?: string;
  affinityState: 'strong' | 'moderate' | 'weak' | 'unknown';
};

export default function StructureViewer({ pdbUrl, affinityState }: Props) {
  const viewerRef = useRef<HTMLDivElement | null>(null);
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
        const color = highlightColor(affinityState);
        plugin.canvas3d?.setProps({ highlightColor: color, backgroundColor: 'white' });
      } catch (error) {
        console.error('Failed to load structure', pdbUrl, error);
      }
    };
    void loadStructure();
    return () => {
      cancelled = true;
    };
  }, [pdbUrl, affinityState]);

  return (
    <Card>
      <div className="section-header">
        <div className="section-row">
          <Eye size={18} aria-hidden="true" />
          <h2>Structure Viewer</h2>
        </div>
        <p>Interactive Fc-FcγR complex with glycan context.</p>
      </div>
      <div className="viewer-frame" style={{ borderColor: highlightColor(affinityState) }}>
        <div ref={viewerRef} className="viewer-canvas" aria-label="Molstar structure viewer" />
        {!pdbUrl && <div className="viewer-placeholder">No structure loaded.</div>}
      </div>
    </Card>
  );
}
