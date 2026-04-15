#!/usr/bin/env python
"""Predict FcγR binding for antibody Fc regions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[2] / "models"))
from esm2_lectin_encoder import ESM2LectinEncoder
from glycan_gnn_encoder import GlycanGNNEncoder
from fcgr_binding_predictor import FcGammaRFcPredictor


GLYCAN_DICT = {
    "G0F": "Glc2Man3GlcNAc4Fuc1",
    "G1F": "Glc2Man3GlcNAc4Fuc1Gal1",
    "G2F": "Glc2Man3GlcNAc4Fuc1Gal2",
    "G0FS": "Glc2Man3GlcNAc4Fuc1Neu5Ac1",
    "G0FB": "Glc2Man3GlcNAc5Fuc1",
    "G0-Fuc": "Glc2Man3GlcNAc4",
}


def load_model(model_path: str, device: str) -> FcGammaRFcPredictor:
    fcgr_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True)
    fc_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True)
    glycan_encoder = GlycanGNNEncoder(embedding_dim=128, hidden_dim=256, num_layers=3)
    model = FcGammaRFcPredictor(fcgr_encoder, fc_encoder, glycan_encoder, hidden_dims=[512, 256, 128])
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def predict_batch(
    model: FcGammaRFcPredictor,
    fcgr_sequences,
    fc_sequences,
    glycan_structures,
    device: str,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(fcgr_sequences, fc_sequences, glycan_structures).to(device)
        return logits.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/fcgr_fc_binding_predictor.pt")
    parser.add_argument("--antibodies", default="data/processed/antibody_fc_regions.csv")
    parser.add_argument("--fcgr-sequences", default="data/reference/fcgr_sequences.csv")
    parser.add_argument(
        "--output",
        default="results/antibody_fcgr_predictions.csv",
    )
    parser.add_argument(
        "--glycan-variants",
        nargs="+",
        default=list(GLYCAN_DICT.keys()),
        help="Glycan variants to predict",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model = load_model(args.model, args.device)
    ab_df = pd.read_csv(args.antibodies)
    fcgr_df = pd.read_csv(args.fcgr_sequences)

    fcgr_dict = dict(zip(fcgr_df["fcgr_name"], fcgr_df["sequence"]))
    output_rows = []

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Predicting FcγR binding for {len(ab_df)} antibodies...")

    for idx, row in ab_df.iterrows():
        ab_id = row["antibody_id"]
        fc_seq = row["fc_sequence"]

        for fcgr_name, fcgr_seq in fcgr_dict.items():
            for glycan_name in args.glycan_variants:
                glycan_struct = GLYCAN_DICT.get(glycan_name)
                if not glycan_struct:
                    continue

                pred_log_kd = float(
                    predict_batch(
                        model,
                        [fcgr_seq],
                        [fc_seq],
                        [glycan_struct],
                        args.device,
                    )[0]
                )
                kd = 10 ** pred_log_kd
                output_rows.append(
                    {
                        "antibody_id": ab_id,
                        "fcgr_name": fcgr_name,
                        "glycan_name": glycan_name,
                        "predicted_log_kd": pred_log_kd,
                        "predicted_kd_nm": kd,
                    }
                )

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(ab_df)} antibodies...")

    pred_df = pd.DataFrame(output_rows)
    pred_df.to_csv(args.output, index=False)
    print(f"\n✓ Predictions saved to {args.output}")
    print(f"  Total predictions: {len(pred_df)}")
    print(f"  Antibodies: {pred_df['antibody_id'].nunique()}")
    print(f"  FcγR variants: {pred_df['fcgr_name'].nunique()}")
    print(f"  Glycan variants: {pred_df['glycan_name'].nunique()}")

    top10 = pred_df.nsmallest(10, "predicted_kd_nm")[
        ["antibody_id", "fcgr_name", "glycan_name", "predicted_kd_nm"]
    ]
    print("\nTop 10 predicted antibody-FcγR combinations (lowest KD):")
    print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
