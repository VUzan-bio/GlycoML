import React from 'react';
import { BarChart3 } from 'lucide-react';
import { PredictionRecord } from '../../types';
import { classifyAffinity } from '../../utils/bindingAnimation';
import Card from './Card';
import styles from './ComparisonTable.module.css';

type Props = {
  data: PredictionRecord[];
  isLoading?: boolean;
};

export default function ComparisonTable({ data, isLoading }: Props) {
  return (
    <Card className={styles.card}>
      <div className={styles.header}>
        <div>
          <h2>Affinity Metrics</h2>
          <p className={styles.subtext}>Measured vs predicted values with ranking.</p>
        </div>
        <BarChart3 size={18} color="#4286F5" aria-hidden="true" />
      </div>
      <div className="table-wrapper">
        <table className="table" aria-label="Binding affinity comparison" aria-busy={isLoading}>
          <thead>
            <tr>
              <th>Glycan Variant</th>
              <th>Measured K<sub>D</sub> (nM)</th>
              <th>Predicted K<sub>D</sub> (nM)</th>
              <th>Binding Free Energy (ΔG)</th>
              <th>Rank</th>
            </tr>
          </thead>
          <tbody>
            {isLoading
              ? Array.from({ length: 3 }).map((_, index) => (
                  <tr key={`skeleton-${index}`}>
                    <td colSpan={5}>
                      <div className={styles.skeletonRow}>
                        <div className="skeleton" />
                        <div className="skeleton" />
                        <div className="skeleton" />
                      </div>
                    </td>
                  </tr>
                ))
              : data.map((row) => {
                  const affinity = classifyAffinity(row.binding_kd_nm);
                  const measured = typeof row.binding_kd_nm === 'number' ? row.binding_kd_nm : null;
                  const predicted =
                    typeof row.predicted_kd_nm === 'number' ? row.predicted_kd_nm : null;
                  const deltaG =
                    typeof row.delta_g_kcal_mol === 'number' ? row.delta_g_kcal_mol : null;
                  return (
                    <tr
                      key={`${row.fcgr_name}-${row.glycan_name}`}
                      className={`affinity-${affinity}`}
                    >
                      <td>{row.glycan_name}</td>
                      <td className="monospace">{measured !== null ? measured.toFixed(2) : 'N/A'}</td>
                      <td className="monospace">
                        {predicted !== null ? predicted.toFixed(2) : 'N/A'}
                      </td>
                      <td className="monospace">
                        {deltaG !== null ? deltaG.toFixed(2) : 'N/A'}
                      </td>
                      <td>
                        {row.affinity_rank ? `${row.affinity_rank} / ${data.length}` : 'N/A'}
                      </td>
                    </tr>
                  );
                })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
