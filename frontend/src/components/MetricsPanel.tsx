import React from 'react';
import { PredictionRecord } from '../types';
import { highlightColor } from '../utils/bindingAnimation';

type Props = {
  record?: PredictionRecord;
};

export default function MetricsPanel({ record }: Props) {
  if (!record) {
    return (
      <div className="panel metrics">
        <div className="panel-header">
          <h2>Binding Metrics</h2>
          <p>Load a glycoform to see quantitative results.</p>
        </div>
      </div>
    );
  }

  const measured = record.binding_kd_nm ?? 0;
  const predicted = record.predicted_kd_nm ?? record.binding_kd_nm ?? 0;
  const deltaG = record.delta_g_kcal_mol ?? 0;
  const rank = record.affinity_rank ?? 0;

  return (
    <div className="panel metrics">
      <div className="panel-header">
        <h2>Binding Metrics</h2>
        <p>Measured vs predicted affinity with derived thermodynamics.</p>
      </div>
      <div className="metrics-grid">
        <div className="metric">
          <span>Measured Kd (nM)</span>
          <strong>{measured.toFixed(2)}</strong>
        </div>
        <div className="metric">
          <span>Predicted Kd (nM)</span>
          <strong>{predicted.toFixed(2)}</strong>
        </div>
        <div className="metric">
          <span>Delta G (kcal/mol)</span>
          <strong>{deltaG.toFixed(2)}</strong>
        </div>
        <div className="metric">
          <span>Affinity rank</span>
          <strong>{rank || 'N/A'}</strong>
        </div>
      </div>
      <div className="affinity-bar">
        <div
          className="affinity-fill"
          style={{
            width: `${Math.min(100, Math.max(5, (1000 / Math.max(1, measured)) * 10))}%`,
            backgroundColor: highlightColor(record.affinity_class || 'unknown'),
          }}
        />
      </div>
    </div>
  );
}
