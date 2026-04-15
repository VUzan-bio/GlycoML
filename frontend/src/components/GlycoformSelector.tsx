import React from 'react';
import { Atom, Hexagon } from 'lucide-react';
import { GlycoformRecord } from '../types';
import Card from './ui/Card';
import Select from './ui/Select';
import styles from './GlycoformSelector.module.css';

type Props = {
  glycoforms: GlycoformRecord[];
  fcgr: string;
  glycan: string;
  onChange: (fcgr: string, glycan: string) => void;
  isLoading?: boolean;
};

export default function GlycoformSelector({
  glycoforms,
  fcgr,
  glycan,
  onChange,
  isLoading,
}: Props) {
  const fcgrOptions = Array.from(new Set(glycoforms.map((g) => g.fcgr_name))).sort();
  const glycanOptions = glycoforms
    .filter((g) => g.fcgr_name === fcgr)
    .map((g) => g.glycan_name)
    .filter((value, idx, arr) => arr.indexOf(value) === idx)
    .sort();

  const fcgrSelectOptions =
    fcgrOptions.length > 0
      ? fcgrOptions.map((option) => ({ label: option, value: option }))
      : [
          {
            label: isLoading ? 'Loading FcγR allotypes...' : 'No FcγR options',
            value: '',
          },
        ];
  const glycanSelectOptions =
    glycanOptions.length > 0
      ? glycanOptions.map((option) => ({ label: option, value: option }))
      : [{ label: isLoading ? 'Loading glycan structures...' : 'No glycan options', value: '' }];

  return (
    <Card className={styles.layout}>
      <div className="section-header">
        <h2>FcγR Allotype</h2>
        <p>Select an FcγR allotype and glycan variant to render.</p>
      </div>
      {isLoading ? (
        <div className={styles.selectGrid} aria-live="polite" aria-busy="true">
          <div className="skeleton" style={{ height: '44px' }} />
          <div className="skeleton" style={{ height: '44px' }} />
        </div>
      ) : (
        <div className={styles.selectGrid}>
          <Select
            id="fcgr-select"
            label="FcγR Allotype"
            value={fcgr}
            options={fcgrSelectOptions}
            onChange={(value) => onChange(value, glycanOptions[0] || '')}
            ariaLabel="Select FcγR allotype"
            disabled={fcgrOptions.length === 0}
          />
          <Select
            id="glycan-select"
            label="Glycan Structure"
            value={glycan}
            options={glycanSelectOptions}
            onChange={(value) => onChange(fcgr, value)}
            ariaLabel="Select glycan structure"
            disabled={glycanOptions.length === 0}
          />
        </div>
      )}
      <div className={styles.helper}>
        <span className={styles.helperItem}>
          <Atom size={14} aria-hidden="true" />
          Updated live with FastAPI predictions.
        </span>
        <span className={styles.helperItem}>
          <Hexagon size={14} aria-hidden="true" />
          Structural variants render from PDB.
        </span>
      </div>
    </Card>
  );
}
