import React, { useEffect, useMemo, useState } from 'react';
import { fetchGlycoforms, fetchPrediction, structureUrl } from '../api';
import { GlycoformRecord, PredictionRecord } from '../types';
import GlycoformSelector from '../components/GlycoformSelector';
import StructureViewer from '../components/StructureViewer';
import MetricsPanel from '../components/MetricsPanel';
import AnimationControls from '../components/AnimationControls';
import ExportButton from '../components/ExportButton';
import { classifyAffinity } from '../utils/bindingAnimation';

export default function SingleView() {
  const [glycoforms, setGlycoforms] = useState<GlycoformRecord[]>([]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycan, setSelectedGlycan] = useState('');
  const [record, setRecord] = useState<PredictionRecord | undefined>(undefined);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGlycoforms()
      .then((data) => {
        setGlycoforms(data);
        if (data.length > 0) {
          setSelectedFcgr(data[0].fcgr_name);
          setSelectedGlycan(data[0].glycan_name);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedFcgr || !selectedGlycan) {
      return;
    }
    fetchPrediction(selectedFcgr, selectedGlycan)
      .then((data) => {
        setRecord({
          ...data,
          affinity_class: data.affinity_class || classifyAffinity(data.binding_kd_nm),
        });
        setError(null);
      })
      .catch((err) => setError(err.message));
  }, [selectedFcgr, selectedGlycan]);

  const affinityState = record?.affinity_class || 'unknown';
  const pdbUrl = record
    ? structureUrl(record.fcgr_name, record.glycan_name, 'pdb')
    : undefined;

  const topGlycoforms = useMemo(() => {
    if (glycoforms.length === 0) {
      return [];
    }
    return [...glycoforms]
      .sort((a, b) => (a.binding_kd_nm || 0) - (b.binding_kd_nm || 0))
      .slice(0, 5);
  }, [glycoforms]);

  const handleSelection = (fcgr: string, glycan: string) => {
    setSelectedFcgr(fcgr);
    setSelectedGlycan(glycan);
  };

  return (
    <div className={`app ${isPlaying ? 'animating' : ''}`}>
      <header className="hero">
        <div>
          <p className="tag">Phase 3 Viewer</p>
          <h1>Fc-Fcgr Binding Explorer</h1>
          <p className="subtitle">
            Visualize glycoform-driven affinity shifts across Fcgr allotypes with
            structural context and quantitative metrics.
          </p>
        </div>
        <div className="hero-card">
          <h3>Quick highlights</h3>
          <ul>
            {topGlycoforms.map((g) => (
              <li key={`${g.fcgr_name}-${g.glycan_name}`}>
                <span>{g.fcgr_name}</span>
                <span>{g.glycan_name}</span>
                <span>{g.binding_kd_nm.toFixed(1)} nM</span>
              </li>
            ))}
          </ul>
        </div>
      </header>

      <main className="content">
        <section className="left">
          <GlycoformSelector
            glycoforms={glycoforms}
            fcgr={selectedFcgr}
            glycan={selectedGlycan}
            onChange={handleSelection}
          />
          <MetricsPanel record={record} />
          <div className="panel actions">
            <div className="panel-header">
              <h2>Exports</h2>
              <p>Download the full Phase 3 prediction table.</p>
            </div>
            <ExportButton disabled={glycoforms.length === 0} />
          </div>
          <AnimationControls isPlaying={isPlaying} onToggle={() => setIsPlaying(!isPlaying)} />
          {error && <div className="panel error">{error}</div>}
        </section>

        <section className="right">
          <StructureViewer pdbUrl={pdbUrl} affinityState={affinityState} />
        </section>
      </main>
    </div>
  );
}
