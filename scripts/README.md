# Scripts

Entrypoints are grouped by purpose.

## scripts/data (ingestion + dataset building)

- `scripts/data/genepix_parser_clean.py`: parse GenePix XLS files using right panel columns.
- `scripts/data/validate_genepix_clean.py`: validate the GenePix parse output.
- `scripts/data/cfg_scraper.py`: parse local CFG array files or download from CFG.
- `scripts/data/phase2_data_downloader.py`: orchestrate UniLectin + CFG downloads.
- `scripts/data/merge_rfu_sources.py`: merge and deduplicate RFU sources.
- `scripts/data/validate_cfg_output.py`: sanity checks for RFU output.
- `scripts/data/build_phase2_cfg_training_from_rfu.py`: build CFG training CSV from RFU + metadata.
- `scripts/data/build_phase2_transfer_dataset.py`: build transfer CSV from JSONL.
- `scripts/data/merge_phase2_datasets.py`: merge CFG + UniLectin for Phase 2.
- `scripts/data/preprocess_glycan_structures.py`: generate glycan features/embeddings.
- `scripts/data/extract_fc_regions.py`: extract Fc regions from antibodies.
- `scripts/data/generate_fcgr_training_data.py`: create FcγR training CSV.

## scripts/train (model training)

- `scripts/train/train_phase2_with_glycan_encoder_export.py`: Phase 2 training + glycan encoder export.
- `scripts/train/train_phase2_deep_learning.py`: deep learning Phase 2 trainer.
- `scripts/train/train_fcgr_binding_model.py`: baseline FcγR regression.
- `scripts/train/train_fcgr_with_transfer_learning.py`: transfer learning for FcγR.

## scripts/analysis (evaluation + prediction)

- `scripts/analysis/evaluate_phase2_model.py`: Phase 2 evaluation and thresholding.
- `scripts/analysis/compare_transfer_learning_results.py`: baseline vs transfer comparison.
- `scripts/analysis/validate_phase2_status.py`: dataset/metrics checks.
- `scripts/analysis/predict_antibody_fcgr_binding.py`: FcγR predictions for antibodies.

## scripts/baselines (legacy baselines)

- `scripts/baselines/train_phase2_glycan_gnn.py`
- `scripts/baselines/train_phase2_lectin_glycan_model.py`
- `scripts/baselines/phase2_baseline_utils.py`
