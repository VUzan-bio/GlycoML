"""Main orchestrator for the glycoengineering data pipeline."""

from __future__ import annotations

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loaders.therasabdab_streamer import TheraSAbDabStreamer
from data_loaders.unilectin_streamer import UniLectinStreamer
from data_pipeline.validator import DataPipelineValidator

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"

DEFAULT_CONFIG: Dict[str, Any] = {
    "phase1": {
        "pdb_ids": None,
        "cache_dir": "./data/pdb_cache",
        "rate_limit": 1.0,
        "query_limit": 500,
    },
    "phase2": {
        "filter_family": None,
        "filter_organism": "Mammalian",
        "cache_dir": "./data/unilectin_cache",
        "rate_limit": 0.2,
    },
    "validation": {
        "output_dir": "./data/validation_reports",
        "min_glycosites_per_ab": 0,
        "min_sequence_length_lectin": 50,
    },
    "logging": {
        "level": "INFO",
        "log_dir": "./logs",
    },
}


def _setup_logger(log_path: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("orchestrator")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(LOG_FORMAT)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class DataPipelineOrchestrator:
    """Coordinate streaming, validation, and export steps."""

    def __init__(self, config_file: str = "config/data_config.yaml") -> None:
        self.config_file = Path(config_file)
        self.config = self._load_config()
        log_level = self.config.get("logging", {}).get("level", "INFO")
        log_dir = Path(self.config.get("logging", {}).get("log_dir", "./logs"))
        self.log_path = log_dir / "orchestrator.log"
        self.logger = _setup_logger(self.log_path, log_level)

    def _load_config(self) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as handle:
                    user_cfg = yaml.safe_load(handle) or {}
                # shallow merge
                for section, values in user_cfg.items():
                    if isinstance(values, dict):
                        cfg.setdefault(section, {}).update(values)
                    else:
                        cfg[section] = values
            except Exception as exc:
                print(f"Failed to load config {self.config_file}: {exc}")
        else:
            # ensure config directory exists with an example for convenience
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as handle:
                yaml.safe_dump(cfg, handle)
        return cfg

    def run_phase1_collection(self) -> Dict[str, int]:
        """Stream antibody structures and write JSONL output."""
        phase1_cfg = self.config.get("phase1", {})
        streamer = TheraSAbDabStreamer(
            pdb_ids=phase1_cfg.get("pdb_ids"),
            cache_dir=phase1_cfg.get("cache_dir", "./data/pdb_cache"),
            rate_limit=phase1_cfg.get("rate_limit", 1.0),
            verbose=True,
            query_limit=phase1_cfg.get("query_limit", 500),
            resume_from_checkpoint=phase1_cfg.get("resume_from_checkpoint", False),
            checkpoint_file=phase1_cfg.get("checkpoint_file"),
        )
        if not streamer.pdb_ids:
            streamer.query_rcsb_antibodies()
        output_path = Path("data") / "phase1_antibodies.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        glyco_total = 0
        processed = set()
        summary_records = []
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pdb_id = record.get("pdb_id")
                    if pdb_id:
                        processed.add(pdb_id)
                        count += 1
                        glyco_total += len(record.get("glycosites", []))
        if processed:
            streamer.pdb_ids = [pid for pid in streamer.pdb_ids if pid not in processed]
        print("Starting Phase 1: Antibody data collection...")
        write_mode = "a" if processed else "w"
        with output_path.open(write_mode, encoding="utf-8") as handle:
            for pdb_id, ab_name, hc_seq, lc_seq, glyco_df in streamer.stream_antibodies():
                if pdb_id in processed:
                    continue
                record = {
                    "pdb_id": pdb_id,
                    "antibody_name": ab_name,
                    "heavy_chain_seq": hc_seq or "",
                    "light_chain_seq": lc_seq or "",
                    "glycosites": glyco_df.to_dict(orient="records"),
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
                glyco_total += len(glyco_df)
                processed.add(pdb_id)
                summary_records.append(
                    {
                        "pdb_id": pdb_id,
                        "antibody_name": ab_name,
                        "hc_length": len(hc_seq or ""),
                        "lc_length": len(lc_seq or ""),
                        "num_glycosites": len(glyco_df),
                    }
                )
                print(f"  Downloaded {pdb_id}... ✓")
        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_path = Path("data") / "antibodies_downloaded.csv"
            summary_df.to_csv(summary_path, mode="a", index=False, header=not summary_path.exists())
        print(f"Phase 1 complete: {count} antibodies, {glyco_total} glycosites identified")
        self.logger.info("Phase 1 complete: %d antibodies, %d glycosites", count, glyco_total)
        return {"antibodies": count, "glycosites": glyco_total}

    def run_phase2_collection(self) -> Dict[str, int]:
        """Stream lectin-glycan pairs and write JSONL output."""
        phase2_cfg = self.config.get("phase2", {})
        streamer = UniLectinStreamer(
            filter_family=phase2_cfg.get("filter_family"),
            filter_organism=phase2_cfg.get("filter_organism"),
            cache_dir=phase2_cfg.get("cache_dir", "./data/unilectin_cache"),
            rate_limit=phase2_cfg.get("rate_limit", 0.2),
            verbose=True,
            max_pairs=phase2_cfg.get("max_pairs"),
        )
        output_path = Path("data") / "phase2_lectin_glycan.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        unique_lectins = set()
        unique_glycans = set()
        processed_pairs = set()
        processed_meta_keys = set()
        summary_records = []
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    lectin_id = record.get("lectin_id")
                    smiles = record.get("glycan_smiles")
                    glytoucan_id = record.get("glytoucan_id") or ""
                    glycan_iupac = record.get("glycan_iupac") or ""
                    pair_key = f"{lectin_id}||{smiles}"
                    if lectin_id:
                        processed_pairs.add(pair_key)
                        processed_meta_keys.add(f"{lectin_id}||{glytoucan_id or glycan_iupac}")
                        count += 1
                        unique_lectins.add(lectin_id)
                        if smiles:
                            unique_glycans.add(smiles)
        print("Starting Phase 2: Lectin-glycan data collection...")
        write_mode = "a" if processed_pairs else "w"
        with output_path.open(write_mode, encoding="utf-8") as handle:
            metadata = streamer.fetch_all_lectins_metadata()
            if processed_meta_keys:
                filtered = []
                for record in metadata:
                    lectin_id = str(record.get("lectin_id") or "").strip()
                    glytoucan_id = str(record.get("glytoucan_id") or "").strip()
                    iupac = str(record.get("iupac") or "").strip()
                    key = f"{lectin_id}||{glytoucan_id or iupac}"
                    if key not in processed_meta_keys:
                        filtered.append(record)
                metadata = filtered
            for (
                lectin_id,
                lectin_name,
                lectin_seq,
                family,
                iupac,
                glycan_smiles,
                gtid,
                kd,
                rfu,
                cls,
                method,
            ) in streamer.stream_pairs_from_records(metadata):
                pair_key = f"{lectin_id}||{glycan_smiles}"
                if pair_key in processed_pairs:
                    continue
                record = {
                    "lectin_id": lectin_id,
                    "lectin_name": lectin_name,
                    "lectin_seq": lectin_seq,
                    "family": family,
                    "glycan_iupac": iupac,
                    "glycan_smiles": glycan_smiles,
                    "glytoucan_id": gtid,
                    "binding_kd_nm": kd,
                    "binding_rfu": rfu,
                    "binding_class": cls,
                    "measured_method": method,
                }
                handle.write(json.dumps(record) + "\n")
                count += 1
                unique_lectins.add(lectin_id)
                if glycan_smiles:
                    unique_glycans.add(glycan_smiles)
                processed_pairs.add(pair_key)
                summary_records.append(
                    {
                        "lectin_id": lectin_id,
                        "lectin_name": lectin_name,
                        "family": family,
                        "glytoucan_id": gtid,
                        "glycan_smiles": glycan_smiles,
                        "binding_class": cls,
                        "kd_nm": kd,
                        "rfu": rfu,
                    }
                )
                if count % 25 == 0:
                    print(f"  Downloaded {count} lectin-glycan pairs...")
        if summary_records:
            summary_path = Path("data") / "lectin_glycan_downloaded.csv"
            pd.DataFrame(summary_records).to_csv(
                summary_path,
                mode="a",
                index=False,
                header=not summary_path.exists(),
            )
        print(
            f"Phase 2 complete: {count} pairs, {len(unique_lectins)} unique lectins, {len(unique_glycans)} unique glycans"
        )
        self.logger.info(
            "Phase 2 complete: %d pairs, %d unique lectins, %d unique glycans",
            count,
            len(unique_lectins),
            len(unique_glycans),
        )
        return {"pairs": count, "lectins": len(unique_lectins), "glycans": len(unique_glycans)}

    def run_validation(self) -> bool:
        """Execute validation across both phases."""
        print("Starting validation...")
        validator = DataPipelineValidator(
            therasabdab_config=self.config.get("phase1", {}),
            unilectin_config=self.config.get("phase2", {}),
            output_dir=self.config.get("validation", {}).get("output_dir", "./data/validation_reports"),
            min_glycosites_per_ab=self.config.get("validation", {}).get("min_glycosites_per_ab", 0),
            min_sequence_length_lectin=self.config.get("validation", {}).get("min_sequence_length_lectin", 50),
        )
        success = validator.run_full_validation()
        print(validator.last_summary)
        return success

    def run_full_pipeline(self) -> bool:
        """Run streaming for both phases followed by validation."""
        print("=" * 50)
        print("GLYCOENGINEERING DATA PIPELINE")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        p1 = self.run_phase1_collection()
        p2 = self.run_phase2_collection()
        success = self.run_validation()

        print("Pipeline summary:")
        print(f"- Antibodies: {p1['antibodies']} entries, {p1['glycosites']} glycosites")
        print(f"- Lectin-glycan pairs: {p2['pairs']} entries")
        print("✓ Validation passed" if success else "✗ Validation reported issues")
        return success


if __name__ == "__main__":
    orchestrator = DataPipelineOrchestrator(config_file="config/data_config.yaml")
    ok = orchestrator.run_full_pipeline()
    if ok:
        print(
            """
✓ Data pipeline completed successfully!

Output files:
- ./data/phase1_antibodies.jsonl
- ./data/phase2_lectin_glycan.jsonl
- ./data/antibodies_validated.csv
- ./data/lectin_glycan_validated.csv
- ./data/validation_reports/summary_*.txt
"""
        )
    else:
        print("✗ Pipeline failed. Check logs in ./logs/")
