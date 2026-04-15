import React, { useMemo, useState } from 'react';
import { AlertCircle, Loader2, Table2, X } from 'lucide-react';
import { BatchPredictLiveResponse, batchPredictLive } from '../api';
import { PredictionRecord } from '../types';
import GlycoformMultiSelector from '../components/GlycoformMultiSelector';
import ComparisonGrid from '../components/ComparisonGrid';
import ComparisonTable from '../components/ui/ComparisonTable';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import { useSearchParams } from 'react-router-dom';
import styles from './CompareView.module.css';

export default function CompareView() {
  const [searchParams] = useSearchParams();
  const initialFcgr = searchParams.get('fcgr') || '';
  const initialGlycans = useMemo(() => {
    const list = searchParams.get('glycans') || searchParams.get('glycan') || '';
    return list ? list.split(',').filter(Boolean) : [];
  }, [searchParams]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycans, setSelectedGlycans] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<PredictionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [structureWarning, setStructureWarning] = useState<string | null>(null);
  const [warningDismissed, setWarningDismissed] = useState(false);
  const [syncRotation, setSyncRotation] = useState(true);
  const [highlightDiffs, setHighlightDiffs] = useState(false);

  const handleCompare = async () => {
    if (!selectedFcgr || selectedGlycans.length < 2) {
      return;
    }
    setIsLoading(true);
    setError(null);
    setStructureWarning(null);
    setWarningDismissed(false);
    try {
      const pairs = selectedGlycans.map((glycan) => ({ fcgr: selectedFcgr, glycan }));
      const response: BatchPredictLiveResponse = await batchPredictLive(pairs);
      setComparisonData(response.results);
      const hasGlycanFlags = response.results
        .map((row) => row.structure?.has_glycan)
        .filter((value) => value !== undefined && value !== null);
      const allTemplateOnly =
        hasGlycanFlags.length > 0 && hasGlycanFlags.every((value) => value === false);
      setStructureWarning(
        response.all_structures_identical || allTemplateOnly
          ? response.structure_warning ||
              'Warning: All structures are rendered from the same Fc-FcγR template without glycan coordinates. Visual differences between glycoforms will not be visible until glycan PDBs are provided in data/structures/glycans/.'
          : null
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Comparison failed';
      setError(message);
      setStructureWarning(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <main className={styles.layout} role="main" id="main-content">
        <aside className={styles.sidebar}>
          <div className={styles.sidebarPanel}>
            <div className="section-title">
              <Table2 size={20} color="#5771FE" aria-hidden="true" />
              Comparative Analysis
            </div>
            <GlycoformMultiSelector
              maxSelections={3}
              onFcgrChange={setSelectedFcgr}
              onGlycansChange={setSelectedGlycans}
              initialFcgr={initialFcgr}
              initialGlycans={initialGlycans}
            />
            <Button
              variant="primary"
              onClick={handleCompare}
              disabled={isLoading || selectedGlycans.length < 2}
              className="predict-button"
              icon={
                isLoading ? (
                  <Loader2 size={16} className="spinner" aria-hidden="true" />
                ) : undefined
              }
            >
              {isLoading ? 'Predicting...' : 'Predict Binding'}
            </Button>
            {error && (
              <div className="error-banner" role="alert">
                <AlertCircle size={16} aria-hidden="true" style={{ marginRight: '8px' }} />
                {error}
              </div>
            )}
          </div>
        </aside>

        <section className={styles.main}>
          {structureWarning && !warningDismissed && (
            <div className="warning-banner" role="status">
              <AlertCircle size={16} aria-hidden="true" style={{ marginRight: '8px' }} />
              <span>{structureWarning}</span>
              <span style={{ marginLeft: 'auto', display: 'inline-flex', gap: '8px' }}>
                <a
                  href="https://github.com/VUzan-bio/GlycoML"
                  target="_blank"
                  rel="noreferrer"
                  className="icon-btn"
                  style={{ width: 'auto', padding: '0 10px' }}
                >
                  Add glycan PDBs
                </a>
                <a
                  href="https://github.com/VUzan-bio/GlycoML"
                  target="_blank"
                  rel="noreferrer"
                  className="icon-btn"
                  style={{ width: 'auto', padding: '0 10px' }}
                >
                  Documentation
                </a>
                <button
                  type="button"
                  className="icon-btn"
                  onClick={() => setWarningDismissed(true)}
                  aria-label="Dismiss warning"
                >
                  <X size={14} aria-hidden="true" />
                </button>
              </span>
            </div>
          )}
          <div className={styles.toolbar}>
            <div className={styles.toggleGroup}>
              <label className={styles.toggleItem}>
                <input
                  type="checkbox"
                  checked={syncRotation}
                  onChange={(event) => setSyncRotation(event.target.checked)}
                />
                Sync rotation
              </label>
              <label className={styles.toggleItem}>
                <input
                  type="checkbox"
                  checked={highlightDiffs}
                  onChange={(event) => setHighlightDiffs(event.target.checked)}
                />
                Highlight differences
              </label>
            </div>
          </div>

          {(isLoading || comparisonData.length > 0) && (
            <>
              <ComparisonTable data={comparisonData} isLoading={isLoading} />
              {comparisonData.length > 0 && (
                <div className={styles.viewerGrid}>
                  <ComparisonGrid
                    fcgrName={selectedFcgr}
                    data={comparisonData}
                    highlightDiffs={highlightDiffs}
                    syncRotation={syncRotation}
                  />
                </div>
              )}
            </>
          )}
          {comparisonData.length === 0 && !isLoading && (
            <Card className={styles.emptyCard}>
              <div className="section-title">No comparison loaded</div>
              <p className="helper-text">Select an FcγR allotype and at least two variants.</p>
            </Card>
          )}
        </section>
      </main>
    </div>
  );
}
