"""Validation utilities for antibody and lectin-glycan data streams."""

from __future__ import annotations

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_loaders.therasabdab_streamer import TheraSAbDabStreamer
from data_loaders.unilectin_streamer import UniLectinStreamer

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("validator")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


class DataPipelineValidator:
    """Validate antibody and lectin-glycan data streams.

    Parameters
    ----------
    therasabdab_config : dict
        Configuration forwarded to :class:`TheraSAbDabStreamer`.
    unilectin_config : dict
        Configuration forwarded to :class:`UniLectinStreamer`.
    output_dir : str, optional
        Directory for validation reports. Defaults to ``./logs/validation``.
    min_glycosites_per_ab : int, default 0
        Minimum expected glycosylation sites per antibody.
    min_sequence_length_lectin : int, default 50
        Minimum length threshold for lectin sequences.
    """

    def __init__(
        self,
        therasabdab_config: Dict,
        unilectin_config: Dict,
        output_dir: str = "./logs/validation",
        min_glycosites_per_ab: int = 0,
        min_sequence_length_lectin: int = 50,
    ) -> None:
        thera_cfg = dict(therasabdab_config)
        thera_cfg["resume_from_checkpoint"] = False
        thera_cfg["checkpoint_file"] = None
        self.antibody_streamer = TheraSAbDabStreamer(**thera_cfg)
        self.lectin_streamer = UniLectinStreamer(**unilectin_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path("logs") / "validator.log"
        self.logger = _setup_logger(self.log_path)
        self.antibody_df: Optional[pd.DataFrame] = None
        self.lectin_df: Optional[pd.DataFrame] = None
        self.min_glycosites_per_ab = min_glycosites_per_ab
        self.min_sequence_length_lectin = min_sequence_length_lectin
        self.last_summary: str = ""

    def validate_antibody_data(self) -> pd.DataFrame:
        """Run validation over antibody stream."""
        records: List[Dict] = []
        site_total = 0
        for pdb_id, ab_name, heavy_seq, light_seq, glycosites in self.antibody_streamer.stream_antibodies():
            hc_len = len(heavy_seq) if heavy_seq else 0
            lc_len = len(light_seq) if light_seq else 0
            issues: List[str] = []
            if hc_len + lc_len < 100:
                issues.append("short_sequence")
            if len(glycosites) < self.min_glycosites_per_ab:
                issues.append("low_glycosite_count")
            for _, row in glycosites.iterrows():
                if not (0 <= float(row.get("plddt", 0)) <= 100):
                    issues.append("invalid_plddt")
                if not (0 <= float(row.get("sasa", 0)) <= 200):
                    issues.append("invalid_sasa")
                motif = str(row.get("motif_type") or "")
                if len(motif) == 3 and (motif[0] != "N" or motif[1] == "P" or motif[2] not in {"S", "T"}):
                    issues.append("invalid_motif")
            site_total += len(glycosites)
            records.append(
                {
                    "pdb_id": pdb_id,
                    "ab_name": ab_name,
                    "hc_length": hc_len,
                    "lc_length": lc_len,
                    "num_glycosites": len(glycosites),
                    "glycosite_positions": ";".join(
                        f"{row.chain}{int(row.position)}" for _, row in glycosites.iterrows()
                    ),
                    "issues": ";".join(sorted(set(issues))) if issues else "",
                }
            )
        df = pd.DataFrame(
            records,
            columns=[
                "pdb_id",
                "ab_name",
                "hc_length",
                "lc_length",
                "num_glycosites",
                "glycosite_positions",
                "issues",
            ],
        )
        self.antibody_df = df
        self.logger.info(
            "Antibody validation complete. Antibodies: %d | Glycosites: %d",
            len(df),
            site_total,
        )
        return df

    def validate_lectin_glycan_data(self) -> pd.DataFrame:
        """Run validation over lectin-glycan pairs."""
        records: List[Dict] = []
        for (
            lectin_id,
            lectin_name,
            lectin_seq,
            family,
            iupac,
            glycan_smiles,
            glytoucan_id,
            kd_nm,
            rfu,
            cls,
            method,
        ) in self.lectin_streamer.stream_pairs():
            issues: List[str] = []
            if len(lectin_seq) <= self.min_sequence_length_lectin:
                issues.append("short_sequence")
            if not glycan_smiles:
                issues.append("missing_smiles")
            if not (0 <= rfu <= 10000):
                issues.append("rfu_out_of_range")
            if cls not in {"strong", "medium", "weak"}:
                issues.append("invalid_class")
            if pd.isna(kd_nm):
                issues.append("missing_kd")
            else:
                kd_molar = kd_nm * 1e-9
                if not (1e-12 <= kd_molar <= 1e-3):
                    issues.append("kd_out_of_range")
            records.append(
                {
                    "lectin_id": lectin_id,
                    "lectin_name": lectin_name,
                    "family": family,
                    "glytoucan_id": glytoucan_id,
                    "glycan_smiles": glycan_smiles,
                    "binding_class": cls,
                    "kd_nm": kd_nm,
                    "rfu": rfu,
                    "method": method,
                    "issues": ";".join(sorted(set(issues))) if issues else "",
                }
            )
        df = pd.DataFrame(
            records,
            columns=[
                "lectin_id",
                "lectin_name",
                "family",
                "glytoucan_id",
                "glycan_smiles",
                "binding_class",
                "kd_nm",
                "rfu",
                "method",
                "issues",
            ],
        )
        if not df.empty:
            duplicates = df[df.duplicated(subset=["lectin_id", "glycan_smiles"], keep=False)]
            if not duplicates.empty:
                self.logger.warning("Detected %d duplicate lectin-glycan pairs", len(duplicates))
        self.lectin_df = df
        self.logger.info("Lectin validation complete. Pairs: %d", len(df))
        return df

    def _collect_issue_codes(self, series: pd.Series) -> List[str]:
        issue_set = set()
        for entry in series.dropna():
            for item in str(entry).split(";"):
                if item:
                    issue_set.add(item)
        return sorted(issue_set)

    def detect_issues(self) -> Dict[str, List[str]]:
        """Aggregate issues discovered during validation."""
        issues: Dict[str, List[str]] = {}
        if self.antibody_df is not None and "issues" in self.antibody_df.columns:
            bad = self.antibody_df[self.antibody_df["issues"] != ""]
            if not bad.empty:
                issues["antibody"] = self._collect_issue_codes(bad["issues"])
        if self.lectin_df is not None and "issues" in self.lectin_df.columns:
            bad = self.lectin_df[self.lectin_df["issues"] != ""]
            if not bad.empty:
                issues["lectin_glycan"] = self._collect_issue_codes(bad["issues"])
        return issues

    def _filter_critical_issues(self, issues: Dict[str, List[str]]) -> Dict[str, List[str]]:
        non_critical = {"missing_kd", "invalid_sasa", "short_sequence", "low_glycosite_count"}
        filtered: Dict[str, List[str]] = {}
        for section, values in issues.items():
            critical = [value for value in values if value not in non_critical]
            if critical:
                filtered[section] = critical
        return filtered

    def generate_summary_report(self) -> str:
        """Create a human-readable validation summary."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = self.output_dir / f"summary_{timestamp}.txt"
        lines = [
            "GLYCOENGINEERING DATA VALIDATION REPORT",
            f"Generated at UTC {timestamp}",
            "",
        ]
        if self.antibody_df is not None:
            valid_antibodies = (self.antibody_df["issues"] == "").sum()
            lines.append(f"Antibodies processed: {len(self.antibody_df)} (valid {valid_antibodies})")
        if self.lectin_df is not None:
            valid_pairs = (self.lectin_df["issues"] == "").sum()
            lines.append(f"Lectin-glycan pairs processed: {len(self.lectin_df)} (valid {valid_pairs})")
        issue_map = self.detect_issues()
        if issue_map:
            lines.append("")
            lines.append("Issues detected:")
            for section, values in issue_map.items():
                lines.append(f"- {section}: {', '.join(values)}")
        else:
            lines.append("")
            lines.append("No critical issues detected.")

        content = "\n".join(lines)
        report_path.write_text(content, encoding="utf-8")
        self.logger.info("Wrote summary report to %s", report_path)
        return content

    def generate_csv_outputs(self) -> None:
        """Persist validated datasets to CSV files."""
        antibodies_path = Path("data") / "interim" / "antibodies_validated.csv"
        lectins_path = Path("data") / "processed" / "lectin_glycan_validated.csv"
        antibodies_path.parent.mkdir(parents=True, exist_ok=True)
        lectins_path.parent.mkdir(parents=True, exist_ok=True)
        if self.antibody_df is not None:
            self.antibody_df.to_csv(antibodies_path, index=False)
        if self.lectin_df is not None:
            self.lectin_df.to_csv(lectins_path, index=False)
        self.logger.info("Saved CSV outputs.")

    def run_full_validation(self) -> bool:
        """Execute the full validation workflow."""
        self.validate_antibody_data()
        self.validate_lectin_glycan_data()
        issues = self.detect_issues()
        self.last_summary = self.generate_summary_report()
        self.generate_csv_outputs()
        return len(self._filter_critical_issues(issues)) == 0


__all__ = ["DataPipelineValidator"]
