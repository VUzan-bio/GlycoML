import React, { useCallback, useEffect, useState } from 'react';
import { fetchGlycoforms, fetchPrediction, structureUrl } from '../api';
import { GlycoformRecord, PredictionRecord } from '../types';
import { AlertCircle } from 'lucide-react';
import ControlsPanel from '../components/ControlsPanel';
import StructuralViewer from '../components/StructuralViewer';
import AffinityMetrics from '../components/AffinityMetrics';
import ExportButton from '../components/ExportButton';
import { classifyAffinity } from '../utils/bindingAnimation';

export default function SingleView() {
  const [glycoforms, setGlycoforms] = useState<GlycoformRecord[]>([]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycan, setSelectedGlycan] = useState('');
  const [record, setRecord] = useState<PredictionRecord | undefined>(undefined);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingGlycoforms, setIsLoadingGlycoforms] = useState(true);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setIsLoadingGlycoforms(true);
    fetchGlycoforms()
      .then((data) => {
        setGlycoforms(data);
        if (data.length > 0) {
          setSelectedFcgr(data[0].fcgr_name);
          setSelectedGlycan(data[0].glycan_name);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setIsLoadingGlycoforms(false));
  }, []);

  const predictBinding = useCallback(() => {
    if (!selectedFcgr || !selectedGlycan) {
      return;
    }
    setIsLoadingPrediction(true);
    fetchPrediction(selectedFcgr, selectedGlycan)
      .then((data) => {
        setRecord({
          ...data,
          affinity_class: data.affinity_class || classifyAffinity(data.binding_kd_nm),
        });
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setIsLoadingPrediction(false));
  }, [selectedFcgr, selectedGlycan]);

  const affinityState = record?.affinity_class || 'unknown';
  const pdbUrl = record
    ? structureUrl(record.fcgr_name, record.glycan_name, 'pdb')
    : undefined;

  const handleSelection = (fcgr: string, glycan: string) => {
    setSelectedFcgr(fcgr);
    setSelectedGlycan(glycan);
    setRecord(undefined);
    setError(null);
  };

  return (
    <div className={`app-shell ${isPlaying ? 'animating' : ''}`}>
      <main className="main-grid" role="main">
        <aside className="sidebar">
          <ControlsPanel
            glycoforms={glycoforms}
            fcgr={selectedFcgr}
            glycan={selectedGlycan}
            onChange={handleSelection}
            onPredict={predictBinding}
            isLoading={isLoadingGlycoforms}
            isPredicting={isLoadingPrediction}
          />
          <div className="sidebar-panel">
            <div className="section-title">Export</div>
            <ExportButton disabled={glycoforms.length === 0} className="full-width" />
            <p className="helper-text" style={{ marginTop: '8px' }}>
              Export structure summaries for downstream analysis.
            </p>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={16} aria-hidden="true" style={{ marginRight: '8px' }} />
              {error}
            </div>
          )}
        </aside>

        <section className="viewer-section">
          <StructuralViewer
            key={pdbUrl || 'empty'}
            pdbUrl={pdbUrl}
            affinityState={affinityState}
            measuredKd={record?.binding_kd_nm}
            isPlaying={isPlaying}
            onToggleAnimation={() => setIsPlaying(!isPlaying)}
            isLoading={isLoadingPrediction}
          />
        </section>
      </main>

      <AffinityMetrics
        record={record}
        isLoading={isLoadingPrediction}
        totalCount={glycoforms.length}
      />
    </div>
  );
}
