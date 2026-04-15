import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchGlycoforms, fetchPrediction, structureUrl } from '../api';
import { GlycoformRecord, PredictionRecord } from '../types';
import { AlertCircle, Clipboard, DownloadCloud, Wand2 } from 'lucide-react';
import ControlsPanel from '../components/ControlsPanel';
import StructuralViewer from '../components/StructuralViewer';
import ExportButton from '../components/ExportButton';
import { classifyAffinity } from '../utils/bindingAnimation';
import { createScoreScale } from '../utils/bindingScore';
import StatCard from '../components/ui/StatCard';
import Button from '../components/ui/Button';
import styles from './PredictView.module.css';

export default function SingleView() {
  const [glycoforms, setGlycoforms] = useState<GlycoformRecord[]>([]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycan, setSelectedGlycan] = useState('');
  const [mode, setMode] = useState<'single' | 'batch'>('single');
  const [record, setRecord] = useState<PredictionRecord | undefined>(undefined);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoadingGlycoforms, setIsLoadingGlycoforms] = useState(true);
  const [isLoadingPrediction, setIsLoadingPrediction] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

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

  const pdbUrl = record
    ? structureUrl(record.fcgr_name, record.glycan_name, 'pdb')
    : undefined;

  const handleSelection = (fcgr: string, glycan: string) => {
    setSelectedFcgr(fcgr);
    setSelectedGlycan(glycan);
    setRecord(undefined);
    setError(null);
  };

  const apiBase = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const apiSnippet = `${apiBase}/api/predict?fcgr=${encodeURIComponent(
    selectedFcgr || 'FcγRIIIA-V158'
  )}&glycan=${encodeURIComponent(selectedGlycan || 'G0F')}`;

  const examplePairs = useMemo(() => {
    const byFcgr = glycoforms.filter((item) => item.fcgr_name === (selectedFcgr || item.fcgr_name));
    return byFcgr.slice(0, 3).map((item) => ({
      fcgr: item.fcgr_name,
      glycan: item.glycan_name,
    }));
  }, [glycoforms, selectedFcgr]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(`curl \"${apiSnippet}\"`);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  const handleExportJson = () => {
    if (!record) {
      return;
    }
    const blob = new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${record.fcgr_name}_${record.glycan_name}_prediction.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const handleExportPymol = () => {
    if (!record) {
      return;
    }
    const key = `${record.fcgr_name}_${record.glycan_name}`.replace(/[^A-Za-z0-9._-]/g, '_');
    const lines = [
      '# GlycoML single-structure PyMOL script',
      'bg_color white',
      `load outputs/phase3_pymol/${key}.pdb, ${key}`,
      `png outputs/phase3_pymol/${key}.png, width=1200, height=1200, dpi=300, ray=1`,
    ];
    const blob = new Blob([lines.join('\\n')], { type: 'text/plain' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${key}.pml`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const affinityState = record?.affinity_class || 'unknown';
  const measured = typeof record?.binding_kd_nm === 'number' ? record.binding_kd_nm : null;
  const predicted =
    typeof record?.predicted_kd_nm === 'number'
      ? record.predicted_kd_nm
      : typeof record?.binding_kd_nm === 'number'
        ? record.binding_kd_nm
        : null;
  const deltaG =
    typeof record?.delta_g_kcal_mol === 'number' ? record.delta_g_kcal_mol : null;
  const affinityLabelMap = {
    strong: 'High Affinity',
    moderate: 'Moderate Affinity',
    weak: 'Low Affinity',
    unknown: 'Affinity Pending',
  } as const;

  const affinityLabel = affinityLabelMap[affinityState];
  const kdValues = useMemo(() => {
    return glycoforms
      .map((row) => row.predicted_kd_nm ?? row.binding_kd_nm)
      .filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  }, [glycoforms]);
  const scoreScale = useMemo(() => createScoreScale(kdValues), [kdValues]);
  const bindingScore = predicted !== null ? scoreScale(predicted) : null;

  return (
    <div className={`app-shell ${isPlaying ? 'animating' : ''}`}>
      <main className={styles.layout} role="main" id="main-content">
        <aside className={styles.sidebar}>
          <ControlsPanel
            glycoforms={glycoforms}
            fcgr={selectedFcgr}
            glycan={selectedGlycan}
            mode={mode}
            onModeChange={setMode}
            onChange={handleSelection}
            onPredict={predictBinding}
            isLoading={isLoadingGlycoforms}
            isPredicting={isLoadingPrediction}
          />
          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={16} aria-hidden="true" style={{ marginRight: '8px' }} />
              {error}
            </div>
          )}
        </aside>

        <section className={styles.results}>
          <div className={styles.resultsHeader}>
            <div>
              <h2>Results</h2>
              <p className={styles.resultsSub}>Live prediction with structural context.</p>
            </div>
            <span className={`status-chip ${affinityState}`}>{affinityLabel}</span>
          </div>

          <div className={styles.statsGrid} aria-busy={isLoadingPrediction}>
            {isLoadingPrediction ? (
              Array.from({ length: 4 }).map((_, index) => (
                <div key={`stat-skeleton-${index}`} className={styles.statSkeleton}>
                  <div className="skeleton" style={{ height: '14px', width: '60%' }} />
                  <div className="skeleton" style={{ height: '24px', width: '40%' }} />
                  <div className="skeleton" style={{ height: '12px', width: '30%' }} />
                </div>
              ))
            ) : (
              <>
                <StatCard
                  label="Binding Probability"
                  value={bindingScore !== null ? bindingScore.toFixed(2) : 'n/a'}
                  tone={affinityState === 'unknown' ? 'neutral' : affinityState}
                  badge="Score (0-1)"
                />
                <StatCard
                  label={
                    <span>
                      Equilibrium Dissociation Constant (K<sub>D</sub>)
                    </span>
                  }
                  value={measured !== null ? `${measured.toFixed(1)} nM` : 'n/a'}
                  tone={affinityState === 'unknown' ? 'neutral' : affinityState}
                  badge="Measured"
                />
                <StatCard
                  label="Binding Free Energy (ΔG)"
                  value={deltaG !== null ? `${deltaG.toFixed(2)} kcal/mol` : 'n/a'}
                  tone="neutral"
                  badge="Predicted"
                />
                <StatCard
                  label="Affinity Rank"
                  value={
                    record?.affinity_rank ? `${record.affinity_rank} / ${glycoforms.length}` : 'n/a'
                  }
                  tone="neutral"
                  badge="Lower is stronger"
                />
              </>
            )}
          </div>

          <div className={styles.heatmapCard}>
            <div className={styles.heatmapHeader}>
              <span>Confidence heatmap</span>
              <span className={styles.heatmapMeta}>Per-residue contribution</span>
            </div>
            <div className={styles.heatmapGrid} aria-busy={isLoadingPrediction}>
              {Array.from({ length: 48 }).map((_, index) => (
                <span
                  key={`heat-${index}`}
                  className={`${styles.heatmapCell} ${isLoadingPrediction ? styles.heatmapCellLoading : ''}`}
                />
              ))}
            </div>
            <div className={styles.heatmapHint}>
              {record
                ? 'Confidence overlays will appear after model updates.'
                : 'Run a prediction to populate residue-level confidence.'}
            </div>
          </div>

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

        <aside className={styles.context}>
          <div className={styles.contextPanel}>
            <div className={styles.contextSection}>
              <div className={styles.contextTitle}>Example Inputs</div>
              <div className={styles.exampleList}>
                {examplePairs.length > 0 ? (
                  examplePairs.map((pair) => (
                    <button
                      key={`${pair.fcgr}-${pair.glycan}`}
                      className={styles.exampleButton}
                      type="button"
                      onClick={() => handleSelection(pair.fcgr, pair.glycan)}
                    >
                      {pair.fcgr} · {pair.glycan}
                    </button>
                  ))
                ) : (
                  <span className="helper-text">Loading example inputs...</span>
                )}
              </div>
            </div>

            <div className={styles.contextSection}>
              <div className={styles.contextTitle}>API Usage</div>
              <pre className={styles.codeBlock}>curl &quot;{apiSnippet}&quot;</pre>
              <Button
                variant="secondary"
                icon={<Clipboard size={16} aria-hidden="true" />}
                onClick={handleCopy}
                className={styles.inlineButton}
              >
                {copied ? 'Copied' : 'Copy command'}
              </Button>
            </div>

            <div className={styles.contextSection}>
              <div className={styles.contextTitle}>Export Options</div>
              <ExportButton disabled={glycoforms.length === 0} className={styles.exportButton} />
              <Button
                variant="secondary"
                icon={<DownloadCloud size={16} aria-hidden="true" />}
                onClick={handleExportJson}
                className={styles.exportButton}
                disabled={!record}
              >
                Export JSON
              </Button>
              <Button
                variant="secondary"
                icon={<Wand2 size={16} aria-hidden="true" />}
                onClick={handleExportPymol}
                className={styles.exportButton}
                disabled={!record}
              >
                PyMOL Script
              </Button>
              <a
                className={styles.downloadLink}
                href={record ? structureUrl(record.fcgr_name, record.glycan_name, 'pdb') : undefined}
                aria-disabled={!record}
              >
                <DownloadCloud size={16} aria-hidden="true" />
                Download PDB
              </a>
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}
