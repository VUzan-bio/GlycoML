"""Evaluate baseline models (SVM, RandomForest)."""

from __future__ import annotations

import argparse
import pickle
from typing import List

import numpy as np

from models.benchmark_models import train_classification_baselines, train_regression_baselines
from models.glycan_encoder import GlycanFingerprintConfig, GlycanFingerprintEncoder
from utils.data_utils import build_label_from_threshold, load_interaction_samples, split_samples
from utils.metrics import accuracy, mae, matthews_corrcoef, mse, pearson
from utils.sequence_features import hashed_kmer_counts


def build_features(samples, fingerprint_bits: int) -> List[List[float]]:
    glycan_encoder = GlycanFingerprintEncoder(
        GlycanFingerprintConfig(n_bits=fingerprint_bits, include_physchem=True, include_iupac=True)
    )
    features = []
    for sample in samples:
        seq_feat = hashed_kmer_counts(sample.lectin_sequence)
        glycan_feat = glycan_encoder.encode(sample.glycan_smiles, sample.glycan_iupac).tolist()
        features.append(seq_feat + glycan_feat)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline models.")
    parser.add_argument("--data", required=True, help="Processed CSV")
    parser.add_argument("--splits", help="Optional pickle with train/val/test indices")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--target", default=None)
    parser.add_argument("--label_threshold", type=float)
    parser.add_argument("--fingerprint_bits", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    samples = load_interaction_samples(args.data)
    if args.splits:
        with open(args.splits, "rb") as handle:
            splits = pickle.load(handle)
        train_samples = [samples[idx] for idx in splits.get("train", [])]
        test_samples = [samples[idx] for idx in splits.get("test", [])]
    else:
        train_samples, _, test_samples = split_samples(samples, seed=args.seed)

    target_key = args.target
    if target_key is None:
        target_key = "rfu_norm" if args.task == "regression" else "label"

    x_train = build_features(train_samples, args.fingerprint_bits)
    x_test = build_features(test_samples, args.fingerprint_bits)

    if args.task == "classification":
        y_train = []
        y_test = []
        for sample in train_samples:
            if sample.label is not None:
                y_train.append(sample.label)
            else:
                value = sample.rfu_norm if sample.rfu_norm is not None else sample.rfu
                if args.label_threshold is None:
                    raise ValueError("label_threshold required when label is missing")
                y_train.append(build_label_from_threshold(value, args.label_threshold))
        for sample in test_samples:
            if sample.label is not None:
                y_test.append(sample.label)
            else:
                value = sample.rfu_norm if sample.rfu_norm is not None else sample.rfu
                if args.label_threshold is None:
                    raise ValueError("label_threshold required when label is missing")
                y_test.append(build_label_from_threshold(value, args.label_threshold))

        models = train_classification_baselines(np.array(x_train), np.array(y_train))
        for name, model in models.items():
            preds = model.predict(np.array(x_test))
            acc = accuracy(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)
            print(f"{name}: acc={acc:.3f} mcc={mcc:.3f}")
    else:
        if target_key == "rfu_norm":
            y_train = [s.rfu_norm if s.rfu_norm is not None else s.rfu for s in train_samples]
            y_test = [s.rfu_norm if s.rfu_norm is not None else s.rfu for s in test_samples]
        else:
            y_train = [s.rfu for s in train_samples]
            y_test = [s.rfu for s in test_samples]
        models = train_regression_baselines(np.array(x_train), np.array(y_train))
        for name, model in models.items():
            preds = model.predict(np.array(x_test))
            print(
                f"{name}: mse={mse(y_test, preds):.4f} mae={mae(y_test, preds):.4f} "
                f"pearson={pearson(y_test, preds):.3f}"
            )


if __name__ == "__main__":
    main()
