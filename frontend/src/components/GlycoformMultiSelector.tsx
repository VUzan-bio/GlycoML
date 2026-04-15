import React, { useEffect, useMemo, useState } from 'react';
import { fetchGlycoforms } from '../api';
import { GlycoformRecord } from '../types';
import Select from './ui/Select';
import styles from './GlycoformMultiSelector.module.css';

interface Props {
  maxSelections: number;
  onFcgrChange: (fcgr: string) => void;
  onGlycansChange: (glycans: string[]) => void;
  initialFcgr?: string;
  initialGlycans?: string[];
}

export default function GlycoformMultiSelector({
  maxSelections,
  onFcgrChange,
  onGlycansChange,
  initialFcgr,
  initialGlycans,
}: Props) {
  const [glycoforms, setGlycoforms] = useState<GlycoformRecord[]>([]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycans, setSelectedGlycans] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);
    fetchGlycoforms()
      .then((data) => {
        setGlycoforms(data);
        if (data.length > 0) {
          const fcgrValue = initialFcgr || data[0].fcgr_name;
          setSelectedFcgr(fcgrValue);
          onFcgrChange(fcgrValue);
          if (initialGlycans && initialGlycans.length > 0) {
            const initialSet = new Set(initialGlycans.slice(0, maxSelections));
            setSelectedGlycans(initialSet);
            onGlycansChange(Array.from(initialSet));
          }
        }
      })
      .finally(() => setIsLoading(false));
  }, [initialFcgr, initialGlycans, maxSelections, onFcgrChange, onGlycansChange]);

  useEffect(() => {
    if (!initialFcgr) {
      return;
    }
    setSelectedFcgr(initialFcgr);
    onFcgrChange(initialFcgr);
  }, [initialFcgr, onFcgrChange]);

  useEffect(() => {
    if (!initialGlycans || initialGlycans.length === 0) {
      return;
    }
    const next = new Set(initialGlycans.slice(0, maxSelections));
    setSelectedGlycans(next);
    onGlycansChange(Array.from(next));
  }, [initialGlycans, maxSelections, onGlycansChange]);

  const fcgrOptions = useMemo(() => {
    return [...new Set(glycoforms.map((g) => g.fcgr_name))].sort();
  }, [glycoforms]);

  const glycanOptions = useMemo(() => {
    return glycoforms
      .filter((g) => g.fcgr_name === selectedFcgr)
      .map((g) => g.glycan_name)
      .sort();
  }, [glycoforms, selectedFcgr]);

  const handleFcgrChange = (value: string) => {
    setSelectedFcgr(value);
    setSelectedGlycans(new Set());
    onFcgrChange(value);
    onGlycansChange([]);
  };

  const handleToggle = (glycan: string) => {
    const next = new Set(selectedGlycans);
    if (next.has(glycan)) {
      next.delete(glycan);
    } else if (next.size < maxSelections) {
      next.add(glycan);
    }
    setSelectedGlycans(next);
    onGlycansChange(Array.from(next));
  };

  const fcgrSelectOptions =
    fcgrOptions.length > 0
      ? fcgrOptions.map((option) => ({ label: option, value: option }))
      : [
          {
            label: isLoading ? 'Loading FcγR allotypes...' : 'No FcγR options',
            value: '',
          },
        ];

  return (
    <div className={styles.container}>
      <Select
        id="fcgr-compare-select"
        label="FcγR Allotype"
        value={selectedFcgr}
        options={fcgrSelectOptions}
        onChange={handleFcgrChange}
        ariaLabel="Select FcγR allotype"
        disabled={isLoading || fcgrOptions.length === 0}
      />

      <div className="section-title">
        Glycan Variants
      </div>

      {isLoading ? (
        <div className={styles.checkboxGrid} aria-busy="true" aria-live="polite">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={`skeleton-${index}`} className="skeleton" style={{ height: '40px' }} />
          ))}
        </div>
      ) : (
        <div className={styles.checkboxGrid} role="group" aria-label="Select glycan structures">
          {glycanOptions.map((glycan) => (
            <label key={glycan} className={styles.checkboxItem}>
              <input
                type="checkbox"
                checked={selectedGlycans.has(glycan)}
                onChange={() => handleToggle(glycan)}
              />
              <span>{glycan}</span>
            </label>
          ))}
        </div>
      )}
      <span className="helper-text">
        Selected {selectedGlycans.size} / {maxSelections}
      </span>
    </div>
  );
}
