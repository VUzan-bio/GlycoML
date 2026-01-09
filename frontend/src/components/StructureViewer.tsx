import React, { useEffect, useRef } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';

import { highlightColor } from '../utils/bindingAnimation';

type Props = {
  pdbUrl?: string;
  affinityState: 'strong' | 'moderate' | 'weak' | 'unknown';
};

export default function StructureViewer({ pdbUrl, affinityState }: Props) {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const pluginRef = useRef<any>(null);

  const ensurePlugin = async () => {
    if (!viewerRef.current) {
      return null;
    }
    if (!pluginRef.current) {
      pluginRef.current = await createPluginUI(viewerRef.current, DefaultPluginUISpec());
    }
    return pluginRef.current;
  };

  useEffect(() => {
    let disposed = false;
    void ensurePlugin().then((plugin) => {
      if (disposed && plugin?.dispose) {
        plugin.dispose();
      }
    });
    return () => {
      disposed = true;
      if (pluginRef.current?.dispose) {
        pluginRef.current.dispose();
      }
      pluginRef.current = null;
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
      const data = await plugin.builders.data.download({ url: pdbUrl }, { state: { isGhost: true } });
      const trajectory = await plugin.builders.structure.parseTrajectory(data, 'pdb');
      await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default');
      if (cancelled) {
        return;
      }
      const color = highlightColor(affinityState);
      plugin.canvas3d?.setProps({ highlightColor: color, backgroundColor: 'white' });
    };
    void loadStructure();
    return () => {
      cancelled = true;
    };
  }, [pdbUrl, affinityState]);

  return (
    <div className="panel viewer">
      <div className="panel-header">
        <h2>Structure Viewer</h2>
        <p>Interactive Fc-Fcgr complex with glycan context.</p>
      </div>
      <div className="viewer-surface" style={{ borderColor: highlightColor(affinityState) }}>
        <div ref={viewerRef} className="viewer-canvas" />
        {!pdbUrl && <div className="viewer-placeholder">No structure loaded.</div>}
      </div>
    </div>
  );
}
