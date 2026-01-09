import React, { useState } from 'react';
import { AlertCircle, Loader2, Table2 } from 'lucide-react';
import { BatchPredictLiveResponse, batchPredictLive } from '../api';
import { PredictionRecord } from '../types';
import GlycoformMultiSelector from '../components/GlycoformMultiSelector';
import ComparisonGrid from '../components/ComparisonGrid';
import ComparisonTable from '../components/ui/ComparisonTable';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';

export default function CompareView() {
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycans, setSelectedGlycans] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<PredictionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [structureWarning, setStructureWarning] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!selectedFcgr || selectedGlycans.length < 2) {
      return;
    }
    setIsLoading(true);
    setError(null);
    setStructureWarning(null);
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
              'Warning: All structures are rendered from the same Fc-FcγR template without glycan coordinates. Visual differences between glycoforms will not be visible until glycan PDBs are provided.'
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
      <main className="main-grid" role="main">
        <aside className="sidebar">
          <div className="sidebar-panel">
            <div className="section-title">
              <Table2 size={20} color="#4286F5" aria-hidden="true" />
              Comparative Analysis
            </div>
            <GlycoformMultiSelector
              maxSelections={3}
              onFcgrChange={setSelectedFcgr}
              onGlycansChange={setSelectedGlycans}
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

        <section className="viewer-section">
          {structureWarning && (
            <div className="warning-banner" role="status">
              <AlertCircle size={16} aria-hidden="true" style={{ marginRight: '8px' }} />
              {structureWarning}
            </div>
          )}
          {(isLoading || comparisonData.length > 0) && (
            <>
              <ComparisonTable data={comparisonData} isLoading={isLoading} />
              {comparisonData.length > 0 && (
                <ComparisonGrid fcgrName={selectedFcgr} data={comparisonData} />
              )}
            </>
          )}
          {comparisonData.length === 0 && !isLoading && (
            <Card>
              <div className="section-title">No comparison loaded</div>
              <p className="helper-text">Select an FcγR allotype and at least two variants.</p>
            </Card>
          )}
        </section>
      </main>
    </div>
  );
}
