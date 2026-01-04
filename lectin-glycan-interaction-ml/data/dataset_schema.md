# Dataset schema (Phase 2)

The scripts expect a CSV with one row per lectin-glycan pair.

Required columns:
- lectin_id: unique lectin identifier
- lectin_sequence: amino acid sequence (single-letter)
- glycan_id: unique glycan identifier
- glycan_smiles: SMILES string for the glycan
- rfu: raw binding signal (RFU)

Optional columns:
- glycan_iupac: IUPAC-carb or composition string (used for monosaccharide counts)
- rfu_norm: normalized RFU value
- label: 0/1 binder label

Example:
```
lectin_id,lectin_sequence,glycan_id,glycan_smiles,glycan_iupac,rfu,rfu_norm,label
lec_001,MASTTTNNW,glc_001,OC[C@H]1O[C@H](O)[C@H](O)[C@H](O)[C@H]1O,Glc,1200,0.45,1
```

The preprocessing script can add rfu_norm and label columns.
