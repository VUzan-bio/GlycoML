# Dataset schema (Phase 1)

Two supported layouts are accepted by the training/prediction scripts.

## Option A: single-chain rows

Columns:
- id: unique sequence identifier
- chain: chain label (H or L or arbitrary)
- sequence: amino acid sequence (single-letter)
- glyco_sites: 1-based positions of known glycosylated Asn residues

Example:
```
id,chain,sequence,glyco_sites
mAb_001,H,EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISSSGSTIYYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCAR,297
mAb_001,L,DIQMTQSPSSLSASVGDRVTITCRASQSIYSSNNHWYQQKPGKAPKLLIYDTSSNLA,60
```

## Option B: heavy/light columns per row

Columns:
- id
- heavy_seq
- light_seq
- heavy_glyco_sites (optional)
- light_glyco_sites (optional)
- glyco_sites (optional, can include H: / L: prefixes, e.g. H:297;L:60)

Example:
```
id,heavy_seq,light_seq,heavy_glyco_sites,light_glyco_sites
mAb_002,EVQLVESGGGLV...,DIQMTQSPSSLS...,297,
```

Notes:
- Motif candidates are derived from N-X-S/T where X != P.
- Positions are parsed as 1-based in the CSV and stored as 0-based internally.

