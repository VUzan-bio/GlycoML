import { GlycoformRecord, PredictionRecord } from './types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export async function fetchGlycoforms(): Promise<GlycoformRecord[]> {
  const res = await fetch(`${API_BASE}/api/glycoforms`);
  if (!res.ok) {
    throw new Error('Failed to fetch glycoforms');
  }
  const data = await res.json();
  return data.glycoforms || [];
}

export async function fetchPrediction(fcgr: string, glycan: string): Promise<PredictionRecord> {
  const url = new URL(`${API_BASE}/api/predict`);
  url.searchParams.set('fcgr', fcgr);
  url.searchParams.set('glycan', glycan);
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error('Prediction not found');
  }
  return res.json();
}

export type BatchPredictLiveResponse = {
  results: PredictionRecord[];
  count: number;
  all_structures_identical?: boolean;
  structure_warning?: string | null;
};

export async function batchPredictLive(
  pairs: Array<{ fcgr: string; glycan: string }>
): Promise<BatchPredictLiveResponse> {
  const res = await fetch(`${API_BASE}/api/batch-predict-live`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(pairs),
  });
  if (!res.ok) {
    throw new Error('Batch prediction failed');
  }
  const data = await res.json();
  return {
    results: data.results || [],
    count: data.count || 0,
    all_structures_identical: data.all_structures_identical || false,
    structure_warning: data.structure_warning || null,
  };
}

export function structureUrl(fcgr: string, glycan: string, format: 'png' | 'pdb' = 'pdb'): string {
  return `${API_BASE}/api/structure/${encodeURIComponent(fcgr)}/${encodeURIComponent(glycan)}?format=${format}`;
}

export function snapshotUrl(fcgr: string, glycan: string): string {
  return `${API_BASE}/api/snapshot?fcgr=${encodeURIComponent(fcgr)}&glycan=${encodeURIComponent(glycan)}`;
}

export function exportUrl(): string {
  return `${API_BASE}/api/export`;
}
