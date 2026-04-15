import React, { useState } from 'react';
import { Layers, Target, Zap } from 'lucide-react';
import { GlycoformRecord } from '../types';
import Button from './ui/Button';
import Select from './ui/Select';
import styles from './ControlsPanel.module.css';
import { Link } from 'react-router-dom';

type Props = {
  glycoforms: GlycoformRecord[];
  fcgr: string;
  glycan: string;
  mode: 'single' | 'batch';
  onChange: (fcgr: string, glycan: string) => void;
  onModeChange: (mode: 'single' | 'batch') => void;
  onPredict: () => void;
  isLoading: boolean;
  isPredicting: boolean;
};

export default function ControlsPanel({
  glycoforms,
  fcgr,
  glycan,
  mode,
  onChange,
  onModeChange,
  onPredict,
  isLoading,
  isPredicting,
}: Props) {
  const [customIupac, setCustomIupac] = useState('');
  const fcgrOptions = Array.from(new Set(glycoforms.map((g) => g.fcgr_name))).sort();
  const glycanOptions = glycoforms
    .filter((g) => g.fcgr_name === fcgr)
    .map((g) => g.glycan_name)
    .filter((value, index, arr) => arr.indexOf(value) === index)
    .sort();

  const fcgrSelectOptions =
    fcgrOptions.length > 0
      ? fcgrOptions.map((option) => ({ label: option, value: option }))
      : [{ label: isLoading ? 'Loading allotypes...' : 'No allotypes found', value: '' }];

  const glycanSelectOptions =
    glycanOptions.length > 0
      ? glycanOptions.map((option) => ({ label: option, value: option }))
      : [{ label: isLoading ? 'Loading variants...' : 'No variants found', value: '' }];

  return (
    <div className={styles.panel}>
      <div className={styles.panelTitle}>Prediction Inputs</div>

      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <Target size={18} color="#5771FE" aria-hidden="true" />
          FcγR Allotype
        </div>
        <Select
          id="fcgr-allotype"
          label="FcγR Allotype"
          value={fcgr}
          options={fcgrSelectOptions}
          onChange={(value) => onChange(value, glycanOptions[0] || '')}
          ariaLabel="Select FcγR allotype"
          disabled={isLoading || fcgrOptions.length === 0}
          hideLabel
        />
        <div className={styles.checkboxRow}>
          <input type="checkbox" checked readOnly disabled aria-label="Use structure template" />
          Use structure (Fc-FcγR template)
        </div>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionHeader}>
          <Layers size={18} color="#5771FE" aria-hidden="true" />
          Glycan Variant
        </div>
        <Select
          id="glycan-variant"
          label="Glycan Variant"
          value={glycan}
          options={glycanSelectOptions}
          onChange={(value) => onChange(fcgr, value)}
          ariaLabel="Select glycan variant"
          disabled={isLoading || glycanOptions.length === 0}
          hideLabel
        />
        <label className={styles.textField}>
          <span className={styles.textLabel}>Custom IUPAC</span>
          <input
            type="text"
            value={customIupac}
            onChange={(event) => setCustomIupac(event.target.value)}
            placeholder="Paste IUPAC sequence"
            disabled={isLoading}
          />
        </label>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionHeader}>Prediction Mode</div>
        <div className={styles.radioGroup} role="radiogroup" aria-label="Prediction mode">
          <label className={styles.radioItem}>
            <input
              type="radio"
              name="prediction-mode"
              checked={mode === 'single'}
              onChange={() => onModeChange('single')}
            />
            Single prediction
          </label>
          <label className={styles.radioItem}>
            <input
              type="radio"
              name="prediction-mode"
              checked={mode === 'batch'}
              onChange={() => onModeChange('batch')}
            />
            Batch (FcγR × glycan grid)
          </label>
        </div>
        {mode === 'batch' && (
          <div className={styles.modeHint}>
            Switch to Compare View to run multi-glycan predictions.{' '}
            <Link to="/compare">Open Compare View</Link>
          </div>
        )}
      </div>

      <Button
        variant="primary"
        onClick={onPredict}
        disabled={isLoading || isPredicting || !fcgr || !glycan}
        className={styles.predictButton}
        icon={<Zap size={18} color="white" aria-hidden="true" />}
      >
        {isPredicting ? 'Predicting...' : 'Predict Binding'}
      </Button>
    </div>
  );
}
