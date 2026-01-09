from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.append(str(SRC_DIR))

from glycoml.phase3.train_fcgr import FcgrRegressor, glycan_composition_vector  # noqa: E402
from glycoml.shared.esm2_embedder import ESM2Embedder  # noqa: E402

LOG = logging.getLogger("glycoml.phase3.inference")


def _safe_torch_load(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _infer_embed_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("fc_proj.weight")
    if weight is None:
        raise ValueError("fc_proj.weight missing in checkpoint.")
    return weight.shape[1]


def _infer_hidden_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("fc_proj.weight")
    if weight is None:
        raise ValueError("fc_proj.weight missing in checkpoint.")
    return weight.shape[0]


def _infer_glycan_dim(state_dict: Dict[str, torch.Tensor]) -> int:
    weight = state_dict.get("glycan_proj.weight")
    if weight is None:
        return len(glycan_composition_vector(""))
    return weight.shape[1]


class Phase3Predictor:
    """Phase 3 Fcgr predictor with cached features."""

    def __init__(self, checkpoint_path: Path, device: Optional[str] = None, esm_model: str = "esm2_t6_8M_UR50D") -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model(checkpoint_path)
        self.embedder = ESM2Embedder(model_name=esm_model, device=self.device)
        self.feature_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _load_model(self, path: Path) -> FcgrRegressor:
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {path}")
        checkpoint = _safe_torch_load(path, self.device)
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise ValueError("Checkpoint does not contain model_state_dict.")

        embed_dim = _infer_embed_dim(state)
        hidden_dim = _infer_hidden_dim(state)
        glycan_dim = _infer_glycan_dim(state)
        use_fcgr = bool(checkpoint.get("use_fcgr", True))
        use_glycan = bool(checkpoint.get("use_glycan", True))

        model = FcgrRegressor(
            embed_dim=embed_dim,
            glycan_dim=glycan_dim,
            hidden_dim=hidden_dim,
            use_fcgr=use_fcgr,
            use_glycan=use_glycan,
        )
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        LOG.info("Loaded Phase3 model from %s (embed_dim=%d, glycan_dim=%d)", path, embed_dim, glycan_dim)
        return model

    def build_features(
        self,
        fcgr_name: str,
        glycan_name: str,
        glycan_structure: str,
        fc_sequence: str,
        fcgr_sequence: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_key = (fcgr_name, glycan_name)
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]

        fc_emb = self.embedder.embed_pooled(fc_sequence)
        fcgr_emb = self.embedder.embed_pooled(fcgr_sequence or fc_sequence)
        glycan_source = glycan_structure or glycan_name
        glycan_vec = glycan_composition_vector(str(glycan_source))
        glycan_tensor = torch.tensor(glycan_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        features = (fc_emb.unsqueeze(0), fcgr_emb.unsqueeze(0), glycan_tensor)
        self.feature_cache[cache_key] = features
        return features

    def predict(
        self,
        fcgr_name: str,
        glycan_name: str,
        glycan_structure: str,
        fc_sequence: str,
        fcgr_sequence: str,
    ) -> Dict[str, float | str | None]:
        try:
            fc_emb, fcgr_emb, glycan_vec = self.build_features(
                fcgr_name, glycan_name, glycan_structure, fc_sequence, fcgr_sequence
            )
        except Exception as exc:
            LOG.warning("Failed to build features for %s/%s: %s", fcgr_name, glycan_name, exc)
            return {
                "predicted_log_kd": float("nan"),
                "predicted_kd_nm": float("nan"),
                "delta_g_kcal_mol": float("nan"),
                "prediction_error": "feature_error",
                "prediction_confidence": 0.0,
            }

        try:
            with torch.no_grad():
                pred = self.model(fc_emb, fcgr_emb, glycan_vec).squeeze().item()
            predicted_kd_nm = float(10 ** pred)
            return {
                "predicted_log_kd": float(pred),
                "predicted_kd_nm": predicted_kd_nm,
                "delta_g_kcal_mol": float(-0.593 * pred),
                "prediction_error": None,
                "prediction_confidence": 0.0,
            }
        except Exception as exc:
            LOG.error("Prediction failed for %s/%s: %s", fcgr_name, glycan_name, exc)
            return {
                "predicted_log_kd": float("nan"),
                "predicted_kd_nm": float("nan"),
                "delta_g_kcal_mol": float("nan"),
                "prediction_error": "model_error",
                "prediction_confidence": 0.0,
            }

    def predict_batch(
        self,
        rows: list[Dict[str, str]],
    ) -> list[Dict[str, float | str | None]]:
        features = []
        for row in rows:
            fcgr_name = row.get("fcgr_name", "")
            glycan_name = row.get("glycan_name", "")
            glycan_structure = row.get("glycan_structure", "") or ""
            fc_sequence = row.get("fc_sequence", "") or ""
            fcgr_sequence = row.get("fcgr_sequence", "") or ""
            features.append(
                self.build_features(fcgr_name, glycan_name, glycan_structure, fc_sequence, fcgr_sequence)
            )

        if not features:
            return []

        fc_emb = torch.cat([item[0] for item in features], dim=0)
        fcgr_emb = torch.cat([item[1] for item in features], dim=0)
        glycan_vec = torch.cat([item[2] for item in features], dim=0)

        with torch.no_grad():
            preds = self.model(fc_emb, fcgr_emb, glycan_vec).squeeze().cpu().numpy()

        results = []
        for pred in np.atleast_1d(preds):
            predicted_kd_nm = float(10 ** float(pred))
            results.append(
                {
                    "predicted_log_kd": float(pred),
                    "predicted_kd_nm": predicted_kd_nm,
                    "delta_g_kcal_mol": float(-0.593 * float(pred)),
                    "prediction_error": None,
                    "prediction_confidence": 0.0,
                }
            )
        return results
