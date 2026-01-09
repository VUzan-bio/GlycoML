import React from 'react';
import { Award, BarChart3, TrendingUp } from 'lucide-react';
import { PredictionRecord } from '../types';
import { classifyAffinity } from '../utils/bindingAnimation';

type Props = {
  record?: PredictionRecord;
  isLoading?: boolean;
  totalCount?: number;
};

const affinityLabelMap = {
  strong: 'High Affinity',
  moderate: 'Moderate Affinity',
  weak: 'Low Affinity',
  unknown: 'Affinity Pending',
} as const;

export default function AffinityMetrics({ record, isLoading, totalCount }: Props) {
  if (isLoading) {
    return (
      <section className="metrics-container" aria-busy="true">
        <div className="metrics-row">
          {Array.from({ length: 3 }).map((_, index) => (
            <div key={`metric-loading-${index}`} className="metric-card">
              <div className="skeleton" style={{ height: '20px', width: '60%' }} />
              <div className="skeleton" style={{ height: '32px', width: '40%', marginTop: '16px' }} />
              <div className="skeleton" style={{ height: '18px', width: '30%', marginTop: '12px' }} />
            </div>
          ))}
        </div>
      </section>
    );
  }

  if (!record) {
    return (
      <section className="metrics-container">
        <div className="metrics-row">
        <div className="metric-card">
          <div className="metric-header">
            <TrendingUp size={20} color="#5771FE" aria-hidden="true" />
            <span>Affinity Metrics</span>
          </div>
          <div className="helper-text">
            Select an FcγR allotype and glycan variant, then Predict Binding.
          </div>
        </div>
      </div>
    </section>
    );
  }

  const measured = typeof record.binding_kd_nm === 'number' ? record.binding_kd_nm : 0;
  const predictedValue =
    typeof record.predicted_kd_nm === 'number'
      ? record.predicted_kd_nm
      : typeof record.binding_kd_nm === 'number'
        ? record.binding_kd_nm
        : null;
  const deltaG = typeof record.delta_g_kcal_mol === 'number' ? record.delta_g_kcal_mol : 0;
  const affinity = record.affinity_class || classifyAffinity(record.binding_kd_nm);
  const affinityLabel = affinityLabelMap[affinity];

  return (
    <section className="metrics-container" aria-label="Affinity metrics">
      <div className="metrics-row">
        <div className="metric-card">
          <div
            className="metric-header"
            title="K_D measured by surface plasmon resonance (Shields et al., J Biol Chem 276:6591-6604, 2001)"
          >
            <TrendingUp size={20} color="#5771FE" aria-hidden="true" />
            <span>
              Equilibrium Dissociation Constant (K<sub>D</sub>)
            </span>
          </div>
          <div className="metric-value">
            <span className="monospace">{measured.toFixed(1)}</span>
            <span>nM</span>
          </div>
          <div className={`metric-badge ${affinity}`}>{affinityLabel}</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <Award size={20} color="#5771FE" aria-hidden="true" />
            <span>Affinity Rank</span>
          </div>
          <div className="metric-value">
            <span className="monospace">{record.affinity_rank ?? 'N/A'}</span>
            <span>{record.affinity_rank && totalCount ? `/ ${totalCount}` : ''}</span>
          </div>
          <div className="metric-badge">Lower is stronger</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <BarChart3 size={20} color="#5771FE" aria-hidden="true" />
            <span>Binding Free Energy (ΔG)</span>
          </div>
          <div className="metric-value">
            <span className="monospace">{deltaG.toFixed(2)}</span>
            <span>kcal/mol</span>
          </div>
          <div className="metric-badge">
            Predicted K_D {predictedValue !== null ? predictedValue.toFixed(1) : 'n/a'} nM
          </div>
        </div>
      </div>
    </section>
  );
}
