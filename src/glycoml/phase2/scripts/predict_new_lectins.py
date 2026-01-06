"""Predict lectin-glycan interactions for new inputs."""

from __future__ import annotations

import argparse
import csv
from typing import List, Tuple

import torch

from ..models.glycan_encoder import GlycanFingerprintConfig, GlycanFingerprintEncoder, GlycanGCNEncoder
from ..models.interaction_module import load_interaction_model
from ..models.protein_encoder import ESM2Embedder, LectinEncoder


def read_inputs(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict lectin-glycan binding for new inputs.")
    parser.add_argument("--model", required=True, help="Path to trained interaction model")
    parser.add_argument("--input_csv", required=True, help="CSV with lectin_sequence and glycan_smiles")
    parser.add_argument("--output_csv", required=True, help="Output CSV with predictions")
    parser.add_argument("--glycan_encoder", choices=["fingerprint", "gcn"], help="Override encoder type")
    parser.add_argument("--fingerprint_bits", type=int, default=2048)
    parser.add_argument("--fingerprint_radius", type=int, default=2)
    parser.add_argument("--model_name", default="esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    meta = checkpoint.get("meta", {})

    glycan_encoder_type = args.glycan_encoder or meta.get("glycan_encoder", "fingerprint")
    fingerprint_bits = meta.get("fingerprint_bits", args.fingerprint_bits)
    fingerprint_radius = meta.get("fingerprint_radius", args.fingerprint_radius)
    model_name = meta.get("model_name", args.model_name)

    model, config = load_interaction_model(args.model, device=device)

    embedder = ESM2Embedder(model_name, device=device)
    lectin_encoder = LectinEncoder(embedder)

    if glycan_encoder_type == "fingerprint":
        glycan_encoder = GlycanFingerprintEncoder(
            GlycanFingerprintConfig(radius=fingerprint_radius, n_bits=fingerprint_bits, include_physchem=True, include_iupac=True)
        )
    else:
        glycan_encoder = GlycanGCNEncoder(out_dim=config.glycan_dim)
        glycan_encoder.to(device)

    rows = read_inputs(args.input_csv)
    outputs: List[dict] = []

    for row in rows:
        sequence = (row.get("lectin_sequence") or row.get("sequence") or "").strip()
        glycan_smiles = (row.get("glycan_smiles") or row.get("smiles") or "").strip()
        glycan_iupac = (row.get("glycan_iupac") or row.get("iupac") or "").strip()
        if not sequence or not glycan_smiles:
            continue
        lectin_emb = lectin_encoder.encode(sequence).unsqueeze(0).to(device)
        if glycan_encoder_type == "fingerprint":
            glycan_emb = glycan_encoder.encode(glycan_smiles, glycan_iupac).unsqueeze(0).to(device)
        else:
            glycan_emb = glycan_encoder([glycan_smiles]).to(device)
        with torch.no_grad():
            pred = model(lectin_emb, glycan_emb).cpu().item()
        row_out = dict(row)
        row_out["prediction"] = f"{pred:.6f}"
        outputs.append(row_out)

    if not outputs:
        raise SystemExit("No valid rows to predict.")

    with open(args.output_csv, "w", newline="") as handle:
        fieldnames = list(outputs[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in outputs:
            writer.writerow(row)

    print(f"Wrote {len(outputs)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
