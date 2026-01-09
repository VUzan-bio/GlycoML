import React, { useState } from 'react';
import { batchPredictLive } from '../api';
import { PredictionRecord } from '../types';
import GlycoformMultiSelector from '../components/GlycoformMultiSelector';
import ComparisonGrid from '../components/ComparisonGrid';
import ComparisonTable from '../components/ComparisonTable';

export default function CompareView() {
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycans, setSelectedGlycans] = useState<string[]>([]);
  const [comparisonData, setComparisonData] = useState<PredictionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCompare = async () => {
    if (!selectedFcgr || selectedGlycans.length < 2) {
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const pairs = selectedGlycans.map((glycan) => ({ fcgr: selectedFcgr, glycan }));
      const results = await batchPredictLive(pairs);
      setComparisonData(results);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Comparison failed';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="hero">
        <div>
          <p className="tag">Phase 3 Compare</p>
          <h1>Side-by-side glycoform comparison</h1>
          <p className="subtitle">
            Compare two to three glycoforms for a single Fcgr allotype. Review
            structure, affinity rank, and predicted delta G in one panel.
          </p>
        </div>
      </header>

      <main className="content">
        <section className="panel">
          <div className="panel-header">
            <h2>Comparison controls</h2>
            <p>Select an Fcgr allotype and 2-3 glycoforms to compare.</p>
          </div>
          <GlycoformMultiSelector
            maxSelections={3}
            onFcgrChange={setSelectedFcgr}
            onGlycansChange={setSelectedGlycans}
          />
          <div className="compare-actions">
            <button
              className="cta"
              onClick={handleCompare}
              disabled={isLoading || selectedGlycans.length < 2}
            >
              {isLoading ? 'Loading...' : 'Compare'}
            </button>
            <span className="helper">
              Selected {selectedGlycans.length}/3 glycoforms
            </span>
          </div>
          {error && <div className="panel error">{error}</div>}
        </section>

        {comparisonData.length > 0 && (
          <section className="panel full">
            <ComparisonTable data={comparisonData} />
            <ComparisonGrid fcgrName={selectedFcgr} data={comparisonData} />
          </section>
        )}
      </main>
    </div>
  );
}
