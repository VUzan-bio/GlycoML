"""FcgammaR binding impact module.

This module exposes the contract for predicting the change in antibody-
FcgammaR binding free energy (Delta Delta G) caused by N-glycan removal or
variant substitution at the conserved Fc N297 site (Arnold et al., Annu. Rev.
Immunol. 2007; Shields et al., J. Biol. Chem. 2002). A trained graph model is
required -- this file does NOT ship one. Silently returning ``delta_g = 0.0``
(as the previous placeholder did) is unsafe because callers may render the
value in a pipeline output or downstream ranking.

A full implementation is available in
``antibody-fc-engineering/models/fcgr_binding_gnn.py`` (``FcDomainGCN``) and
must be trained on experimental SPR affinity data (Bruhns et al., Blood 2009;
Dekkers et al., Front. Immunol. 2017; Shields et al. 2001) before inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import warnings

import torch


@dataclass
class FcgrPrediction:
    delta_g: float
    note: str


class FcgrBindingPredictor:
    """Thin loader that refuses to silently produce zeroes."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.model: Optional[torch.nn.Module] = None
        self._model_path = model_path

        if model_path is None:
            return
        try:
            loaded = torch.load(model_path, map_location=self.device)
        except Exception as exc:
            warnings.warn(f"Failed to load FcgammaR model from '{model_path}': {exc}")
            return
        # Accept either a state_dict (requires external architecture wiring) or
        # a full pickled nn.Module.
        if isinstance(loaded, torch.nn.Module):
            self.model = loaded.to(self.device)
            self.model.eval()
        else:
            warnings.warn(
                "FcgammaR checkpoint was not an nn.Module; refusing to use a "
                "raw state_dict without an explicit architecture."
            )

    def predict_delta_g(self, sequence: str, glyco_sites: Iterable[int]) -> FcgrPrediction:
        """Predict the FcgammaR binding free-energy change.

        Raises:
            NotImplementedError: when no trained model has been loaded. The
                previous placeholder returned ``delta_g=0.0`` which is
                observationally indistinguishable from a neutral prediction and
                silently corrupts downstream analyses.
        """
        if self.model is None:
            raise NotImplementedError(
                "FcgrBindingPredictor has no trained model. Integrate the "
                "FcDomainGCN from antibody-fc-engineering/models/fcgr_binding_gnn.py "
                "and provide a weights checkpoint via `model_path=...`. See the "
                "module docstring for recommended training data (Bruhns 2009, "
                "Dekkers 2017, Shields 2001)."
            )
        raise NotImplementedError(
            "FcgrBindingPredictor.predict_delta_g graph construction is not "
            "wired to the loaded model. Use the predictor in "
            "antibody-fc-engineering/models/fcgr_binding_gnn.py which "
            "consumes a 3D structure directly."
        )

