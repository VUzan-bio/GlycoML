export type AffinityState = 'strong' | 'moderate' | 'weak' | 'unknown';

export function classifyAffinity(kdNm?: number): AffinityState {
  if (kdNm === undefined || Number.isNaN(kdNm)) {
    return 'unknown';
  }
  if (kdNm < 100) {
    return 'strong';
  }
  if (kdNm > 1000) {
    return 'weak';
  }
  return 'moderate';
}

export function highlightColor(state: AffinityState): string {
  switch (state) {
    case 'strong':
      return '#2d8a58';
    case 'weak':
      return '#b4443b';
    case 'moderate':
      return '#d6a24a';
    default:
      return '#64748b';
  }
}
