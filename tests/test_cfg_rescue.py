import pandas as pd
import pytest

from scripts.parsers import cfg_xlsx_parser
from scripts import merge_rfu_sources


def test_cfg_xlsx_parser_basic(tmp_path):
    pytest.importorskip("openpyxl")

    meta = pd.DataFrame(
        {
            "experiment_id": [1234],
            "array_version": ["PA_v5"],
            "investigator": ["Test PI"],
            "sample_name": ["ConA"],
        }
    )

    data = pd.DataFrame(
        {
            "GlycanID": [1, 2],
            "RFU": [1500, 3000],
            "Normalized": [40, 80],
            "StDev": [10, 20],
            "%CV": [1.0, 2.0],
        }
    )

    xlsx_path = tmp_path / "cfg_1234.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame({"Key": ["Experiment ID"], "Value": [1234]}).to_excel(writer, index=False, sheet_name="Summary")
        data.to_excel(writer, index=False, sheet_name="RFU Data")

    parsed = cfg_xlsx_parser.parse_single_xlsx(xlsx_path, meta, rfu_threshold=2000.0)
    assert not parsed.empty
    assert set(["experiment_id", "glycan_id", "rfu_raw", "rfu_normalized"]).issubset(parsed.columns)
    assert parsed["experiment_id"].iloc[0] == 1234


def test_deduplicate_priority():
    df = pd.DataFrame(
        {
            "experiment_id": [1, 1],
            "array_version": ["PA_v5", "PA_v5"],
            "glycan_id": [10, 10],
            "cfg_glycan_iupac": ["Glc", "Glc"],
            "glytoucan_id": ["G1", "G1"],
            "lectin_sample_name": ["ConA", "ConA"],
            "rfu_raw": [3000, 1000],
            "rfu_normalized": [80, 20],
            "normalization_method": ["cfg_xlsx", "api"],
            "stdev": [0.0, 0.0],
            "cv": [0.0, 0.0],
            "investigator": ["PI", "PI"],
            "data_source": ["CFG_manual", "GlycoPattern"],
            "conclusive": [True, False],
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z"],
        }
    )

    deduped = merge_rfu_sources.deduplicate_rfus(df)
    assert len(deduped) == 1
    assert deduped.iloc[0]["data_source"] == "CFG_manual"
