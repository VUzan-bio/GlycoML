import React from 'react';
import { GlycoformRecord } from '../types';

type Props = {
  glycoforms: GlycoformRecord[];
  fcgr: string;
  glycan: string;
  onChange: (fcgr: string, glycan: string) => void;
};

export default function GlycoformSelector({ glycoforms, fcgr, glycan, onChange }: Props) {
  const fcgrOptions = Array.from(new Set(glycoforms.map((g) => g.fcgr_name))).sort();
  const glycanOptions = glycoforms
    .filter((g) => g.fcgr_name === fcgr)
    .map((g) => g.glycan_name)
    .filter((value, idx, arr) => arr.indexOf(value) === idx)
    .sort();

  return (
    <div className="panel selector">
      <div className="panel-header">
        <h2>Fcgr and Glycan</h2>
        <p>Select a receptor allotype and glycoform to load the structure.</p>
      </div>
      <div className="selector-grid">
        <label className="field">
          <span>Fcgr allotype</span>
          <select
            value={fcgr}
            onChange={(event) => onChange(event.target.value, glycanOptions[0] || '')}
          >
            {fcgrOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Glycan</span>
          <select value={glycan} onChange={(event) => onChange(fcgr, event.target.value)}>
            {glycanOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
