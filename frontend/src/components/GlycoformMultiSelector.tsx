import React, { useEffect, useMemo, useState } from 'react';
import { fetchGlycoforms } from '../api';
import { GlycoformRecord } from '../types';

interface Props {
  maxSelections: number;
  onFcgrChange: (fcgr: string) => void;
  onGlycansChange: (glycans: string[]) => void;
}

export default function GlycoformMultiSelector({
  maxSelections,
  onFcgrChange,
  onGlycansChange,
}: Props) {
  const [glycoforms, setGlycoforms] = useState<GlycoformRecord[]>([]);
  const [selectedFcgr, setSelectedFcgr] = useState('');
  const [selectedGlycans, setSelectedGlycans] = useState<Set<string>>(new Set());

  useEffect(() => {
    fetchGlycoforms().then((data) => {
      setGlycoforms(data);
      if (data.length > 0) {
        setSelectedFcgr(data[0].fcgr_name);
        onFcgrChange(data[0].fcgr_name);
      }
    });
  }, [onFcgrChange]);

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

  return (
    <div className="selector-grid">
      <label className="field">
        Fcgr allotype
        <select value={selectedFcgr} onChange={(e) => handleFcgrChange(e.target.value)}>
          {fcgrOptions.map((fcgr) => (
            <option key={fcgr} value={fcgr}>
              {fcgr}
            </option>
          ))}
        </select>
      </label>

      <div className="field">
        <span>Select glycoforms (2-3)</span>
        <div className="checkbox-grid">
          {glycanOptions.map((glycan) => (
            <label key={glycan} className="checkbox-item">
              <input
                type="checkbox"
                checked={selectedGlycans.has(glycan)}
                onChange={() => handleToggle(glycan)}
              />
              <span>{glycan}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}
