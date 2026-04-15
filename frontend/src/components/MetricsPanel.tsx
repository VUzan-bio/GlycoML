import React from 'react';
import { Activity, Atom, FlaskConical, Hexagon, Box } from 'lucide-react';
import { PredictionRecord } from '../types';
import { classifyAffinity } from '../utils/bindingAnimation';
import Card from './ui/Card';
import StatCard from './ui/StatCard';

type Props = {
  record?: PredictionRecord;
  isLoading?: boolean;
};

export default function MetricsPanel({ record, isLoading }: Props) {
  if (isLoading) {
    return (
      <Card>
        <div className="section-header">
          <h2>Binding Affinity</h2>
          <p>Loading affinity metrics...</p>
        </div>
        <div className="metric-grid">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={`metric-skeleton-${index}`} className="skeleton" style={{ height: '92px' }} />
          ))}
        </div>
      </Card>
    );
  }

  if (!record) {
    return (
      <Card>
        <div className="section-header">
          <h2>Binding Affinity</h2>
          <p>Select a glycan structure to review affinity metrics.</p>
        </div>
      </Card>
    );
  }

  const measured = record.binding_kd_nm ?? 0;
  const predicted = record.predicted_kd_nm ?? record.binding_kd_nm ?? 0;
  const deltaG = record.delta_g_kcal_mol ?? 0;
  const rank = record.affinity_rank ?? 0;
  const affinity = record.affinity_class || classifyAffinity(record.binding_kd_nm);
  const tone = affinity === 'unknown' ? 'neutral' : affinity;

  return (
    <Card>
      <div className="section-header">
        <div className="section-row">
          <Activity size={18} aria-hidden="true" />
          <h2>Binding Affinity</h2>
          <span className={`badge badge-${affinity}`}>{affinity}</span>
        </div>
        <p>Measured vs predicted Kd with derived thermodynamics.</p>
      </div>
      <div className="metric-grid">
        <StatCard
          label="Measured Kd (nM)"
          value={measured.toFixed(2)}
          icon={<Atom size={16} aria-hidden="true" />}
          badge={affinity}
          tone={tone}
        />
        <StatCard
          label="Predicted Kd (nM)"
          value={predicted.toFixed(2)}
          icon={<FlaskConical size={16} aria-hidden="true" />}
          badge={affinity}
          tone={tone}
        />
        <StatCard
          label="Delta G (kcal/mol)"
          value={deltaG.toFixed(2)}
          icon={<Hexagon size={16} aria-hidden="true" />}
          badge={affinity}
          tone={tone}
        />
        <StatCard
          label="Affinity Rank"
          value={rank ? `${rank}` : 'N/A'}
          subtitle={rank ? 'Lower is stronger' : undefined}
          icon={<Box size={16} aria-hidden="true" />}
          badge={affinity}
          tone={tone}
        />
      </div>
    </Card>
  );
}
