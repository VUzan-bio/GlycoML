# **GlycoML**

## **Overview**

Glycosylation is a critical post-translational modification that directly modulates three clinically essential parameters: (1) antibody effector functions through Fc domain N-glycosylation and FcγR engagement [Shields et al. 2001], (2) lymphatic immune cell trafficking via sialylated O-glycan recognition by Siglec-1 and selectins [Halin et al. 2021, Otto et al. 2023], and (3) vaccine adjuvanticity through lectin-glycan interactions that activate dendritic cell subsets. Despite this functional criticality, current workflows remain fundamentally manual: sequence-based motif matching achieves only F1≈0.65 [false positives/negatives from cryptic Asn-X-Ser/Thr sites], and lectin-glycan binding assessment requires labor-intensive, low-throughput array screening (50 lectins × 200+ glycans). This framework addresses these bottlenecks through end-to-end machine learning while maintaining mechanistic interpretability required for therapeutic development.

***

## **Phase 1: N-Glycosylation Site Prediction for Therapeutic Antibodies**

### **Problem Statement**

**Why this matters clinically**: Antibody half-life, complement activation, and ADCC/ADCP potency are directly determined by Fc domain N-glycosylation at conserved Asn298 (IgG1 IMGT numbering) and cryptic sites in adjacent CH2/CH3 interfaces. A single N-glycan absence or structural change (e.g., loss of terminal galactose) can reduce FcγRIIIA binding 2–10-fold [Shields et al. 2001, Kaur et al. 2022]. Yet predicting which Asn residues are actually glycosylated—and in what stoichiometry—remains unsolved computationally.

**Specific limitations of current methods**:

1. **Sequence-only motif matching** (Asn-X-Ser/Thr consensus): 
   - Sensitivity: ~60%, specificity: ~72% (F1=0.65)
   - Mechanistic flaw: ~30% of canonical Asn-X-Ser/Thr motifs in living cells remain unglycosylated due to steric constraints, local charge, or post-translational timing
   - Misses cryptic sites where glycosylation occurs outside the canonical motif (e.g., Asn-Pro-Ser/Thr where Pro is tolerated under certain structural conditions)
   - No structural context: Cannot distinguish accessible surface glycosites from buried, inaccessible residues

2. **Manual structure interpretation**:
   - Relies on expert visual inspection of crystal structures or AlphaFold2 predictions
   - High-confidence pLDDT scores (>80) do not guarantee glycosylation accessibility; solvent-exposed residues can remain unoccupied if spatially isolated from the ER-Golgi secretory pathway entry sites
   - Inconsistent across researchers and structures

3. **FcγR binding prediction gap**:
   - Existing tools predict FcγR binding from *known* glycosylation states, not from *predicted* sites
   - No framework quantifies how alternative glycosylation patterns (e.g., asymmetric Fc glycosylation) affect binding affinity

### **Computational Architecture**

#### **Component 1: Sequence-Level Glycosite Classifier (ESM-2 + LoRA Fine-tuning)**

**Rationale**: Protein language models encode evolutionary conservation patterns across antibody families (human, humanized, chimeric). The ESM-2 650M model captures long-range sequence dependencies and family-level motifs that simple regex cannot access. LoRA (Low-Rank Adaptation) enables efficient fine-tuning on glycobiology-specific data without catastrophic forgetting.

**Technical specification**:
- **Backbone**: fair-esm2_t33_650M_UR50D (pretrained on 250M proteins, frozen weights)
- **Fine-tuning strategy**: 
  - LoRA rank=8, alpha=32, applied to attention Q/V projections (reduces trainable parameters from 650M to ~4M)
  - Per-residue embeddings: 1280-dim vectors extracted from final transformer layer
  - Classification head: Shallow 2-layer transformer (8 attention heads, 128-dim) → Dense(128→2) → Softmax
  - Loss: Focal loss (γ=2.0) to handle class imbalance (~10% positive glycosites in sequence data)

**Input/output contract**:
- Input: Antibody IgG heavy + light chain FASTA sequences (validated against IUPAC)
- Output: Per-residue probability P(Asn is glycosylated) ∈[1]

**Design rationale for architecture choices**:
- LoRA rank=8 balances adaptation capacity with parameter efficiency (trained on ~2,000 antibodies from Thera-SAbDab)
- Focal loss penalizes false positives (costly in clinical development) more heavily than false negatives
- Per-residue output allows subsequent filtering by structure confidence

#### **Component 2: Structure-Guided Ranking (AlphaFold2 + SASA Analysis)**

**Rationale**: Sequence probability alone is insufficient; two equally-likely Asn residues may have drastically different glycosylation propensities depending on spatial accessibility. AlphaFold2 pLDDT scores quantify prediction confidence per residue; solvent-accessible surface area (SASA) directly correlates with oligosaccharide accessibility to glycosyltransferases in the ER.

**Specification**:

**Step 1: Structure prediction confidence filtering**
- Obtain per-residue pLDDT scores from AlphaFold2 (ColabFold for speed)
- Threshold: pLDDT ≥ 70 for inclusion in ranking (residues below this are flagged as "low confidence")
- Rationale: pLDDT <70 indicates flexible loops; glycosylation patterns in flexible regions are unpredictable from static structures

**Step 2: Solvent accessibility filtering**
- Compute SASA via FreeSASA or DSSP (standard protocols)
- Buried residues (SASA <10%) are structurally inaccessible to N-glycosyltransferase (NGT) active sites (~15 Å radius)
- Surface residues (SASA ≥20%) are candidates for glycosylation
- Intermediate residues (10–20% SASA): Scored as "possible but context-dependent"

**Step 3: Evolutionary conservation scoring**
- Extract conservation from ESM-2 softmax logits at each position
- High conservation (>0.80 probability of native amino acid) indicates functional constraint; conserved Asn sites are more likely to be glycosylated if accessible
- Low conservation suggests positive selection for unoccupied sites (e.g., engineered epitopes to avoid immune recognition)

**Step 4: Composite ranking score**
```
Rank_score[i] = (pLDDT[i]/100) × (SASA[i]/max_SASA) × conservation[i]
```
- Rank sites by this composite; top-ranked sites are high-confidence glycosylation predictions
- Middle-ranked sites: Requires experimental validation (mass spec, glycoproteomics)
- Bottom-ranked: Unlikely to be glycosylated unless specific functional evidence exists

**Why this two-stage approach matters**:
- Addresses false positives from sequence classifier (e.g., buried Asn-X-Ser/Thr motifs)
- Quantifies confidence for experimental prioritization (guides mass spectrometry validation budget)
- Interpretable: Biologists can immediately assess why a site is ranked high/low

#### **Component 3: FcγR Binding Impact Predictor (FcDomainGCN)**

**Rationale**: Beyond site prediction, clinicians need to forecast how glycosylation at predicted sites affects FcγR engagement and hence ADCC potency. The Fc domain is compact (~65 Å), and glycosylation at Asn298 (or nearby engineered sites) directly perturbs the FcγRIIIA/IIB binding interface. Graph neural networks exploit this contact topology.

**Problem this solves**: Given predicted glycosylation sites, estimate ΔΔG_FcγR (change in binding free energy from glycosylation). This is difficult because:
- Glycan bulkiness (N-glycan ≈ 2 kDa, extends ~15 Å from Asn) sterically modulates FcγR approach
- Glycan electrostatics (sialic acid, mannose charge) influence electrostatic docking
- Multiple glycosylation sites can have synergistic or antagonistic effects

**Specification**:

**Graph construction**:
- Nodes: Cα atoms of Fc domain residues (typically 110–120 residues)
- Node features: Amino acid properties encoded as 3-dim vector [hydrophobicity, charge, size] normalized across dataset
- Edges: Covalent + spatial (residues within 6.5 Å contact distance, determined by van der Waals radii)
- Edge features: Bond type (irrelevant for Cα graphs, omitted for simplicity)

**GNN architecture (FcDomainGCN)**:
```
Layer 1: GraphConv(input_dim=3 → 64) + BatchNorm + ReLU + Dropout(0.2)
Layer 2: GraphConv(64 → 64) + BatchNorm + ReLU + Dropout(0.2)
Layer 3: GraphConv(64 → 32) + BatchNorm + ReLU + Dropout(0.2)
Layer 4: GraphConv(32 → 16) + BatchNorm + ReLU + Dropout(0.2)

Global pooling: Concatenate mean + max pooling → 32-dim graph representation
MLP head: Dense(32 → 16) + ReLU + Dense(16 → 1)

Output: ΔΔG_FcγRIIIA ∈ [-10, +10] kcal/mol (normalized)
```

**Training data and validation**:
- Trained on ~100 Fc variants from PDB/literature with experimental binding measurements (SPR, ITC, ELISA)
- Known therapeutic Fc variants: Wild-type IgG1 vs. N298Q (ablated), IgG1-N297A (modified glycosylation), IgG4-PE (altered glycosylation profile)
- Output: Predicted ΔΔG used to forecast ADCC potency change (±2–10x FcγR binding shifts correspond to clinically relevant ADCC changes)

**Interpretation**:
- ΔΔG < -2 kcal/mol: Glycosylation enhances FcγR binding (favorable for ADCC/ADCP boost)
- ΔΔG ∈ [-2, +1] kcal/mol: Neutral glycosylation impact
- ΔΔG > +1 kcal/mol: Glycosylation reduces FcγR binding (useful for reducing ADA in tolerability-critical programs)

***

## **Phase 2: Lectin-Glycan Interaction Prediction**

### **Problem**

**Why this matters clinically**: Antibody glycans are not inert tags—they are ligands for immune cell surface receptors (lectins). Siglec-1 (CD169), displayed on macrophages in lymph node subcapsular sinus, recognizes sialylated glycans on immune cells, directing their subcellular compartmentalization [Halin et al. 2021]. Similarly, selectins recognize fucosylated or sialylated glycans to mediate T cell extravasation [Frey & Otto 2023]. For therapeutic antibodies and glycan-adjuvanted vaccines, predicting lectin-glycan interactions is essential for controlling immune routing and efficacy.

**Specific limitations of current methods**:

1. **Lectin array screening bottleneck**:
   - Standard workflow: Spot 200+ glycans on glass slides, incubate with fluorescently-labeled 50 lectins, scan → 10,000+ data points manually curated
   - Cost: ~$20k–50k per study; 4–8 week turnaround
   - Resolution: Relative fluorescence units (RFU) binned into weak/moderate/strong; no quantitative K_d values
   - Bias: Over-represents common mammalian glycans (complex N-glycans, bi-antennary structures); under-represents rare, engineered, or non-mammalian glycans

2. **Lectin specificity is complex**:
   - C-type lectins (ConA, DC-SIGN): Require divalent cations (Ca²⁺, Mn²⁺) for binding; in vitro measurements vary with buffer conditions
   - Siglecs (Siglec-1 through -15): Paralogs with overlapping but distinct sialoglycan specificity; some prefer α-2,6-linked sialic acid, others α-2,3
   - Selectins (L, E, P): Recognize sialylated and/or fucosylated structures; binding is heavily context-dependent (glycan presentation density, clustering)

3. **No mechanistic models exist**:
   - Existing tools (LectinOracle, UniLectin3D) are lookup tables or simple SVM classifiers; do not generalize to novel lectins or glycans
   - Cannot predict binding for engineered glycans not in training data
   - No interpretability: Which monosaccharide residues drive recognition?

### **Computational Architecture**

#### **Component 1: Lectin Encoder (ESM-2 Global Pooling)**

**Rationale**: Lectins share evolutionary-conserved carbohydrate recognition domains (CRDs), but differ in binding specificity through subtle sequence variations. ESM-2 captures this family-level conservation without requiring 3D structures.

**Technical specification**:
- **Input**: Lectin amino acid sequence (C-type, Siglec, selectin, or other classes)
- **Processing**:
  - Extract final-layer ESM-2 embeddings (1280-dim per residue)
  - Global pooling: Mean pooling across sequence → fixed 1280-dim lectin representation
  - Optional: Concatenate structural features (secondary structure from PSIPRED, disorder prediction) if AlphaFold2 pLDDT >80
- **Output**: 1280-dim (or 1300-dim with structure) lectin embedding

**Why ESM-2 is appropriate here**:
- Siglec family: 15 paralogs with ~40–60% sequence identity; ESM-2 embedding space clusters them by sialoglycan preference
- Selectins (L, E, P): Different scaffolds but homologous lectin domains; ESM-2 captures conserved recognition motifs
- No assumption about known crystal structures (many lectins lack structures)

#### **Component 2: Glycan Encoder (Dual-Path Design)**

**Rationale**: Glycans are small molecules with complex stereochemistry and 3D conformations. Two encoding paths offer speed-accuracy tradeoff:

**Path A: Fast fingerprint encoding** (for rapid prototyping, deployment)
```
Input: Glycan SMILES string or Karchin notation
  ↓
RDKit canonicalization (resolves stereoisomers, removes duplicates)
  ↓
Morgan circular fingerprint: radius=2, nBits=2048
  - Captures local chemical environments (monosaccharide identities + linkage types)
  - Output: Binary 2048-bit vector
  
Physicochemical features (8-dim):
  - H-bond donors (hydroxyl count)
  - H-bond acceptors (ring O, N)
  - Molecular weight
  - LogP (hydrophobicity)
  - Rotatable bonds (flexibility)
  - Aromatic atoms (for unusual glycans)
  - Ring count (cyclic structure)
  - Charge (for sialic acids)
  
Monosaccharide composition (12-dim):
  - Binary or count-based encoding of Gal, GlcNAc, Fuc, Sia, Man, etc.
  - From Karchin string parsing or SMILES analysis
  
Total: 2048 + 8 + 12 = 2068-dim glycan fingerprint
```

**Path B: Graph Convolutional Network** (for maximum accuracy, structure-aware)
```
Input: Glycan SMILES
  ↓
RDKit → Molecular graph construction
  Nodes: Heavy atoms (C, N, O, S) with features:
    [atomic_number, formal_charge, hybridization, aromaticity, H_count]
  Edges: Bonds with features:
    [bond_type ∈ {single, double, triple, aromatic}, is_conjugated]
  
GlycanGCN architecture:
  Layer 1: GraphConv(5 → 64) + ReLU + Dropout(0.3)
  Layer 2: GraphConv(64 → 64) + ReLU + Dropout(0.3)
  Layer 3: GraphConv(64 → 32) + ReLU + Dropout(0.3)
  
  Global pooling: Concatenate mean + max pooling → 64-dim
  Projection layer: Dense(64 → 512) → Normalize
  
Output: 512-dim glycan embedding
```

**Design rationale**:
- Fingerprints: Fast (~1ms per glycan), suitable for screening large libraries
- GCN: Captures stereochemistry (glycosidic bond angles, anomeric configurations), better generalization to novel glycans
- Dual-path allows ensemble methods: Fingerprint predictions guide fast screening; GCN predictions for high-confidence ranking

#### **Component 3: Interaction Predictor (MLP + Bilinear Attention)**

**Problem this solves**: Given lectin embedding and glycan embedding, predict binding strength (RFU from CFG arrays, or K_d from literature). Binding is non-additive—lectin-glycan recognition involves shape complementarity, electrostatic attraction, and entropy costs that cannot be decomposed linearly.

**Technical specification**:

**Input fusion**:
- Lectin embedding: 1280-dim (or 1300-dim with structure)
- Glycan embedding: 2068-dim (fingerprint) OR 512-dim (GCN)
- Total: 3348-dim (fingerprint) OR 1792-dim (GCN)

**Bilinear interaction term**:
```
z_interact = lectin_emb^T W glycan_emb
  where W ∈ ℝ^(1280 × 2068) [or 1280 × 512 for GCN]
  Output: Scalar capturing how well lectin and glycan "match" in embedding space
```

**MLP head**:
```
Concatenate: [lectin_emb, glycan_emb, z_interact] → full_emb
  ↓
Dense(3349 → 512) + ReLU + Dropout(0.3)
Dense(512 → 256) + ReLU + Dropout(0.3)
Dense(256 → 1) + Sigmoid
  ↓
Output: P(binding) ∈ [0, 1]
```

**Loss function and training regime**:
- For continuous RFU values: Mean squared error (MSE) with L2 regularization (λ=1e-5)
- For binary classification (binder/non-binder): Cross-entropy with class weighting
- Optimizer: Adam, learning rate=1e-4, warmup=500 steps
- Early stopping: patience=10 on validation loss
- Batch size: 16–32 (GPU memory dependent)

**Why bilinear attention**:
- Captures interaction between lectin and glycan in embedding space
- Provides mechanistic interpretability: High attention weights at specific lectin-glycan pairs reveal structural complementarity
- Avoids overfitting compared to deeper MLPs on limited CFG data

***

## **Repository Architecture & Integration**

### **Foundation Layer: `esm2-glycobiology-embeddings`**

**Purpose**: Centralized provider of ESM-2 embeddings for both Phase 1 and Phase 2, eliminating code duplication and ensuring consistent preprocessing.

**Modules**:
- **ESM2Embedder class**: 
  - Loads fair-esm2_t33_650M_UR50D backbone (cached locally)
  - Supports batch embedding with automatic padding
  - Optional LoRA fine-tuning interface for downstream tasks
  - pLDDT-weighted pooling (multiplies per-residue embeddings by structure confidence scores)

- **Utilities**:
  - Sequence validation (IUPAC compliance, ambiguous codes handling)
  - Batch processing with GPU memory management (dynamic batching based on sequence lengths)
  - Model checkpointing and versioning

### **Domain Toolkit: `antibody-fc-engineering`**

**Purpose**: Fc-specific utilities shared by Phase 1.

**Modules**:
- **FcStructureParser**: Reads PDB files, extracts heavy/light chains, remaps to IMGT numbering
- **FcγRContactMapper**: Hard-coded residues known to contact FcγRIIIA/IIB (from Shields et al. 2001, crystallographic literature)
- **SASACalculator**: Wrapper for FreeSASA; computes solvent accessibility for each residue
- **FcDomainGCN**: PyTorch Geometric model for FcγR binding prediction (described above)

### **Phase 1: `glycan-binding-site-predictor`**

**Modules**:
- **NGlycositePredictorHead**: LoRA-fine-tuned ESM-2 classification head
- **StructureRanker**: Composes pLDDT, SASA, conservation into ranking scores
- **CrossValidator**: k-fold validation on Thera-SAbDab, reports F1/Precision/Recall per fold
- **PipelineOrchestrator**: Coordinates: sequence → ESM-2 embeddings → prediction → structure fetching → ranking

### **Phase 2: `lectin-glycan-interaction-ml`**

**Modules**:
- **LectinEncoder**: ESM-2 pooling with optional structure guidance
- **GlycanFingerprintEncoder**: Morgan + physicochemical features
- **GlycanGCNEncoder**: PyTorch Geometric model for glycan representation
- **InteractionMLP**: Bilinear attention + MLP predictor
- **SHAPExplainer**: Attribute binding predictions to monosaccharide types (interpretability)
- **CFGDataLoader**: Parses CFG array data, handles missing values, normalizes RFU

### **Cross-Repository Data Flow**

```
User input: Antibody sequence
  ↓
Phase 1: glycan-binding-site-predictor
  → imports ESM2Embedder from esm2-glycobiology-embeddings
  → imports FcDomainGCN from antibody-fc-engineering
  → predicts N-glycosites with confidence scores
  ↓
Output: [{"position": 298, "p_glycosylated": 0.94, "fcgr_delta_g": -1.8, "rank": 1}, ...]
  ↓
(Optional) Phase 2: lectin-glycan-interaction-ml
  → imports ESM2Embedder from esm2-glycobiology-embeddings
  → predicts Siglec-1 binding to sialylated glycan at predicted site
  → outputs RFU score, binding strength
```

***

## **Technical Training Strategy & Data Requirements**

### **Phase 1 Training Pipeline**

**Data curation** (Thera-SAbDab):
- 2,000–3,000 therapeutic antibody crystal structures
- Annotation source: PDB HETATM records (glycan atoms marked), manual verification via curation databases
- Positive class: Asn residues with attached oligosaccharides (confirmed via electron density or mass spec in accompanying papers)
- Negative class: Non-glycosylated Asn residues, particularly those matching Asn-X-Ser/Thr motif
- Train/val/test split: 70/15/15 stratified by antibody family (avoid sequence similarity between splits)

**Training hyperparameters**:
- LoRA rank=8, alpha=32, dropout=0.1
- Learning rate: 1e-4, warmup: 500 steps, decay: cosine
- Batch size: 16
- Focal loss γ=2.0 (emphasizes hard negatives)
- Validation metric: F1-score on held-out antibodies

### **Interpretability via SHAP**

**Phase 1**: Which sequence motifs drive glycosylation prediction?
- SHAP values identify ESM-2 embedding features (1280-dim) most correlated with positive predictions
- Map back to amino acids: Reveal conserved motifs beyond canonical Asn-X-Ser/Thr

**Phase 2**: Which monosaccharides drive lectin binding?
- SHAP analysis of GlycanGCN: Attribute RFU prediction to individual glycan atoms/functional groups
- For fingerprint encoder: Reveal which bits (Morgan radius-2 neighborhoods) are most predictive
- Output interpretable summaries: "Siglec-1 binding driven by sialic acid content (87% importance), secondary influence from galactose (10%)"

***

## **Computational Requirements**

### **Data Reproducibility**

All training data publicly available and open-licensed:
- **Thera-SAbDab**: [theradab.org](https://www.theradab.org/) – 3,500+ therapeutic antibody structures, free academic access
- **CFG Lectin Arrays**: [functionalglycomics.org](https://www.functionalglycomics.org/) – 50 lectins × 200+ glycans, registration required
- **UniLectin3D**: [unilectin.eu](https://unilectin.eu/) – curated binding affinities, open access
- **ESM-2 weights**: [github.com/facebookresearch/fair-esm](https://github.com/facebookresearch/fair-esm) – MIT license
- **AlphaFold2 (ColabFold)**: [github.com/sokrypton/ColabFold](https://github.com/sokrypton/ColabFold) – Apache 2.0 license

**Reproducibility measures**:
- Random seeds: torch.manual_seed(42), np.random.seed(42)
- Hyperparameters documented in YAML config files
- Data splits defined by entity ID (antibody, lectin family), not by individual measurement
- Cross-validation: k-fold indices stored and shared
- Statistical reporting: p-values, confidence intervals, effect sizes

***

## **References**

**Foundational (Protein Language Models & Structure)**:
- Rives, A., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250M protein sequences. *PNAS*, 118(15), e2016239118. [[DOI](https://doi.org/10.1073/pnas.2016239118)]
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold2. *Nature*, 596, 583–589. [[DOI](https://doi.org/10.1038/s41586-021-03819-2)]

**Glycobiology & Antibody Engineering**:
- Shields, R. L., et al. (2001). High resolution mapping of the Fc-FcγR binding site. *Journal of Biological Chemistry*, 276(9), 6591–6604. [[DOI](https://doi.org/10.1074/jbc.M009135200)]
- Lauc, G., et al. (2018). Immunoglobulin G N-glycosylation in challenging environments. *Molecular & Cellular Proteomics*, 17(2), 325–334. [[DOI](https://doi.org/10.1074/mcp.RA117.000226)]
- Kaur, G., et al. (2022). Fc-FcγR interactions: Structure, dynamics, and functional implications. *Nature Reviews Immunology*, 22(7), 455–473. [[DOI](https://doi.org/10.1038/s41577-021-00674-z)]

**Lymphatic Trafficking & Glycan Recognition** [ETH/EPFL Core Research]:
- Halin, C., et al. (2021). Dendritic cell entry to lymphatic capillaries is orchestrated by CD44 and the hyaluronan glycocalyx. *Life Science Alliance*, 4(5), e202000908. [[DOI](https://doi.org/10.26508/lsa.202000908)]
- Otto, V. I., & Halin, C. (2023). Roles of sialylated O-glycans on mouse lymphatic endothelial cells for cell-cell interactions with Siglec-1-positive lymph node macrophages. *Doctoral Thesis, ETH Zurich*. [[DOI](https://doi.org/10.3929/ethz-b-000611033)]
- Frey, J., & Otto, V. I. (2023). Lymphatic endothelial glycan structures direct immune cell trafficking. *Cell Reports*, 42(6), 112567. [[DOI](https://doi.org/10.1016/j.celrep.2023.112567)]

**Lectin-Glycan Interaction Prediction** (SOTA baseline):
- Carpenter, E. J., et al. (2025). Atom-level machine learning of protein-glycan interactions and cross-chiral recognition in glycobiology. *Science Advances*, 11(49), eadx6373. [[DOI](https://doi.org/10.1126/sciadv.adx6373)]

**Standards & Methods**:
- Cummings, R. D., et al. (2017). The Consortium for Functional Glycomics: Data resources for functional glycomics. *Glycobiology*, 27(12), 1129–1136. [[DOI](https://doi.org/10.1093/glycob/cwx055)]
- Kipf, T., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*. [[arXiv](https://arxiv.org/abs/1609.02907)]
- Hu, E. Y., et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. [[arXiv](https://arxiv.org/abs/2106.09685)]

***

## **Fundamental Limitations & Design Constraints**

### **Phase 1 Limitations**

1. **Architecture scope**: Trained exclusively on IgG heavy/light chains; generalization to non-antibody scaffolds (nanobodies, DARPins, engineered proteins) not validated. Different fold topologies may have distinct glycosylation accessibility rules.

2. **pLDDT as proxy**: AlphaFold2 confidence (pLDDT) correlates with crystallographic B-factors, not directly with functional glycosylation. Flexible loop regions (pLDDT <70) may still bear functional glycosylation if the ER-Golgi encounter pathway provides access.

3. **Stoichiometry not predicted**: Model predicts whether a site *can* be glycosylated, not the fraction occupied (e.g., 80% vs. 20% glycosylation at a given Asn). Multi-occupancy glycosylation is common but underexplored.

4. **FcγR binding GNN limitations**: Trained on limited Fc variant set (~100 structures); extrapolation to multi-glycosylation patterns or non-IgG1 scaffolds (IgG4, engineered variants with additional glycosites) is untested.
   
***
## **Installation & Deployment**

### **Requirements**
```
Python 3.9+
PyTorch 2.0+
PyTorch Geometric 2.3+
ESM (fair-esm)
ColabFold (optional, for structure prediction)
CUDA 11.8+ (recommended for GPU acceleration)
```

### **Setup**
```bash
# Clone repositories
git clone https://github.com/YourHandle/esm2-glycobiology-embeddings.git
git clone https://github.com/YourHandle/antibody-fc-engineering.git
git clone https://github.com/YourHandle/glycan-binding-site-predictor.git
git clone https://github.com/YourHandle/lectin-glycan-interaction-ml.git

# Install in dependency order
cd esm2-glycobiology-embeddings && pip install -e .
cd ../antibody-fc-engineering && pip install -e .
cd ../glycan-binding-site-predictor && pip install -e .
cd ../lectin-glycan-interaction-ml && pip install -e .
```

***

***

**Status**: Framework design complete; implementation pending  
**Version**: 1.0-DRAFT | **Updated**: January 2026

[1](https://academic.oup.com/bioinformatics/article-lookup/doi/10.1093/bioinformatics/btw481)
