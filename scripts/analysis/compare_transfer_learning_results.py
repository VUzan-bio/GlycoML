#!/usr/bin/env python
"""Compare baseline FcγR model vs transfer learning variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch


def load_model_metrics(checkpoint_path: str) -> Dict[str, Optional[float]]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        "test_mse": checkpoint.get("test_mse", checkpoint.get("val_mse")),
        "epoch": checkpoint.get("epoch"),
        "transfer": checkpoint.get("transfer_learning", False),
        "glycan_frozen": checkpoint.get("glycan_encoder_frozen"),
    }


def main() -> None:
    models = {
        "Baseline (no transfer)": "models/fcgr_fc_binding_predictor.pt",
        "Transfer (frozen glycan)": "models/fcgr_transfer_frozen.pt",
        "Transfer (fine-tuned glycan)": "models/fcgr_transfer_finetuned.pt",
    }

    print("=" * 80)
    print("TRANSFER LEARNING COMPARISON")
    print("=" * 80)

    results = []
    for name, path in models.items():
        if Path(path).exists():
            metrics = load_model_metrics(path)
            results.append(
                {
                    "Model": name,
                    "Test MSE": metrics["test_mse"],
                    "Transfer": metrics["transfer"],
                    "Glycan Frozen": metrics["glycan_frozen"],
                    "Epoch": metrics["epoch"],
                }
            )
            print(f"\n{name}:")
            print(f"  Test MSE: {metrics['test_mse']}")
            print(f"  Trained for: {metrics['epoch']} epochs")
        else:
            print(f"\n{name}: NOT FOUND")

    if results and results[0]["Test MSE"] is not None:
        baseline_mse = results[0]["Test MSE"]
        for res in results[1:]:
            if res["Test MSE"] is None:
                continue
            improvement = 100 * (baseline_mse - res["Test MSE"]) / baseline_mse
            print(f"\n{res['Model']} improvement: {improvement:.1f}%")

    print("\n" + "=" * 80)
    df = pd.DataFrame(results)
    output = Path("results/transfer_learning_comparison.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"✓ Saved to {output}")


if __name__ == "__main__":
    main()
