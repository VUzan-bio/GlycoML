import React from 'react';
import { Layers, Target, Zap } from 'lucide-react';
import { GlycoformRecord } from '../types';
import Button from './ui/Button';
import Select from './ui/Select';

type Props = {
  glycoforms: GlycoformRecord[];
  fcgr: string;
  glycan: string;
  onChange: (fcgr: string, glycan: string) => void;
  onPredict: () => void;
  isLoading: boolean;
  isPredicting: boolean;
};

export default function ControlsPanel({
  glycoforms,
  fcgr,
  glycan,
  onChange,
  onPredict,
  isLoading,
  isPredicting,
}: Props) {
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
    <div className="sidebar-panel">
      <div className="control-section">
        <div className="section-title">
          <Target size={20} color="#4286F5" aria-hidden="true" />
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
      </div>

      <div className="control-section">
        <div className="section-title">
          <Layers size={20} color="#4286F5" aria-hidden="true" />
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
      </div>

      <Button
        variant="primary"
        onClick={onPredict}
        disabled={isLoading || isPredicting || !fcgr || !glycan}
        className="predict-button"
        icon={
          <Zap size={20} color="white" aria-hidden="true" />
        }
      >
        {isPredicting ? 'Predicting...' : 'Predict Binding'}
      </Button>
    </div>
  );
}
