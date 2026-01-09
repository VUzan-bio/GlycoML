import React, { useEffect, useRef } from 'react';
import { createPluginUI } from 'molstar/lib/mol-plugin-ui';
import { PluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import 'molstar/lib/mol-plugin-ui/skin/light.scss';

import { highlightColor } from '../utils/bindingAnimation';

type Props = {
  pdbUrl?: string;
  affinityState: 'strong' | 'moderate' | 'weak' | 'unknown';
};

export default function StructureViewer({ pdbUrl, affinityState }: Props) {
  const viewerRef = useRef<HTMLDivElement | null>(null);
  const pluginRef = useRef<any>(null);

  useEffect(() => {
    if (!viewerRef.current || pluginRef.current) {
      return;
    }
    const spec: PluginUISpec = {
      ...DefaultPluginUISpec(),
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      layoutShowPluginState: false,
      layoutShowTaskQueue: false,
    };
    pluginRef.current = createPluginUI(viewerRef.current, spec);
  }, []);

  useEffect(() => {
    if (!pluginRef.current) {
      return;
    }
    const plugin = pluginRef.current;
    if (!pdbUrl) {
      plugin.clear();
      return;
    }
    plugin.clear();
    plugin.loadStructureFromUrl(pdbUrl, 'pdb').then(() => {
      const color = highlightColor(affinityState);
      plugin.canvas3d?.setProps({ highlightColor: color });
    });
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
