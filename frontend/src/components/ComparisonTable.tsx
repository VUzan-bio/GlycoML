import React from 'react';
import { PredictionRecord } from '../types';
import { classifyAffinity } from '../utils/bindingAnimation';

interface Props {
  data: PredictionRecord[];
}

export default function ComparisonTable({ data }: Props) {
  return (
    <div className="comparison-table">
      <h3>Binding metrics</h3>
      <table>
        <thead>
          <tr>
            <th>Glycoform</th>
            <th>Measured Kd (nM)</th>
            <th>Predicted Kd (nM)</th>
            <th>Delta G (kcal/mol)</th>
            <th>Rank</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const affinity = classifyAffinity(row.binding_kd_nm);
            return (
              <tr key={`${row.fcgr_name}-${row.glycan_name}`} className={`affinity-${affinity}`}>
                <td>{row.glycan_name}</td>
                <td>{row.binding_kd_nm.toFixed(2)}</td>
                <td>{row.predicted_kd_nm ? row.predicted_kd_nm.toFixed(2) : 'N/A'}</td>
                <td>{row.delta_g_kcal_mol ? row.delta_g_kcal_mol.toFixed(2) : 'N/A'}</td>
                <td>{row.affinity_rank ? `${row.affinity_rank} / ${data.length}` : 'N/A'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
