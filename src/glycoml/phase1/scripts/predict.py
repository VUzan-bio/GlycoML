"""Predict candidate N-glycosylation sites."""

from __future__ import annotations

import argparse
import csv
from typing import Dict, Iterable, List, Tuple

import torch

from ..models.esm2_classifier import ESM2Embedder, extract_motif_embedding, load_classifier
from ..models.structure_ranker import parse_plddt_from_pdb, load_sasa_from_csv, rank_sites
from ..models.fcgr_binding_module import FcgrBindingPredictor
from ..utils.sequence import find_nglyco_motifs


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks: List[str] = []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header:
            records.append((header, "".join(seq_chunks)))
    return records


def build_predictions(
    embedder: ESM2Embedder,
    classifier,
    sequence: str,
    device: torch.device,
) -> List[Dict[str, object]]:
    motif_positions = find_nglyco_motifs(sequence)
    if not motif_positions:
        return []
    residue_embeddings = embedder.embed_sequence(sequence)
    features = [extract_motif_embedding(residue_embeddings, pos) for pos in motif_positions]
    x = torch.stack(features).to(device)
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.sigmoid(logits).cpu().tolist()
    preds = []
    for pos, prob in zip(motif_positions, probs):
        preds.append(
            {
                "position": pos + 1,
                "motif": sequence[pos : pos + 3],
                "probability": prob,
            }
        )
    preds.sort(key=lambda item: item["probability"], reverse=True)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict N-glycosylation sites.")
    parser.add_argument("--model", required=True, help="Path to trained classifier checkpoint")
    parser.add_argument("--sequence", help="Single sequence input")
    parser.add_argument("--fasta", help="FASTA file with sequences")
    parser.add_argument("--pdb", help="Optional AlphaFold PDB for structure ranking")
    parser.add_argument("--sasa_csv", help="Optional SASA CSV (chain, position, sasa)")
    parser.add_argument("--chain_id", help="Chain ID for PDB/SASA filtering")
    parser.add_argument("--out_csv", help="Write predictions to CSV")
    parser.add_argument("--fcgr_model", help="Optional Fcgr GNN checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.sequence and not args.fasta:
        raise SystemExit("Provide --sequence or --fasta")

    device = torch.device(args.device)
    classifier, config = load_classifier(args.model, device=device)
    embedder = ESM2Embedder(config.model_name, device=device)

    inputs: List[Tuple[str, str]] = []
    if args.sequence:
        inputs.append(("query", args.sequence.strip()))
    if args.fasta:
        inputs.extend(read_fasta(args.fasta))

    fcgr_predictor = FcgrBindingPredictor(args.fcgr_model, device=device) if args.fcgr_model else None

    all_outputs: List[Dict[str, object]] = []
    for name, seq in inputs:
        preds = build_predictions(embedder, classifier, seq, device=device)
        if args.pdb:
            plddt = parse_plddt_from_pdb(args.pdb, chain_id=args.chain_id)
            sasa = load_sasa_from_csv(args.sasa_csv, chain_id=args.chain_id) if args.sasa_csv else None
            ranked = rank_sites([p["position"] - 1 for p in preds], plddt, sasa_scores=sasa)
            ranked_positions = {item.position: item for item in ranked}
            for pred in preds:
                rank_info = ranked_positions.get(pred["position"] - 1)
                if rank_info:
                    pred["structure_score"] = rank_info.score
                    pred["plddt"] = rank_info.plddt
                    pred["sasa"] = rank_info.sasa

        if fcgr_predictor:
            glyco_positions = [pred["position"] - 1 for pred in preds if pred["probability"] >= 0.5]
            fcgr = fcgr_predictor.predict_delta_g(seq, glyco_positions)
            for pred in preds:
                pred["fcgr_delta_g"] = fcgr.delta_g
                pred["fcgr_note"] = fcgr.note

        for pred in preds:
            pred["sequence_id"] = name
            all_outputs.append(pred)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as handle:
            fieldnames = list(all_outputs[0].keys()) if all_outputs else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_outputs:
                writer.writerow(row)
    else:
        for row in all_outputs:
            print(row)


if __name__ == "__main__":
    main()

