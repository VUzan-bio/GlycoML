"""ESM2 embedder utilities for glycobiology workflows.

Features:
- Cached inference with CPU/GPU auto-detection.
- Optional LoRA fine-tuning via PEFT.
- Batch processing with sequence validation.
- pLDDT-guided confidence weighting.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import esm
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install fair-esm: `pip install fair-esm`") from exc

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ESM2Embedder:
    """Production ESM2 wrapper for glycobiology tasks."""

    MAX_SEQ_LENGTH = {
        "esm2_t6_8M_UR50D": 2048,
        "esm2_t12_35M_UR50D": 2048,
        "esm2_t30_150M_UR50D": 2048,
        "esm2_t33_650M_UR50D": 1024,
        "esm2_t48_15B_UR50D": 512,
    }

    PLDDT_THRESHOLDS = {
        "very_high": 90,
        "high": 80,
        "moderate": 70,
        "low": 50,
    }

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        cache_size: int = 128,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        if cache_dir:
            os.environ.setdefault("TORCH_HOME", cache_dir)
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        logger.info("Initializing ESM2 (%s) on %s", model_name, self.device)
        self._load_model()
        if use_lora:
            self._setup_lora()

    def _load_model(self) -> None:
        try:
            loader = getattr(esm.pretrained, "load_model_and_alphabet", None)
            if loader is None:
                raise RuntimeError("esm.pretrained.load_model_and_alphabet not available")
            self.model, self.alphabet = loader(self.model_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to load ESM2 model '{self.model_name}': {exc}") from exc

        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = getattr(self.model, "embed_dim", 1280)

        for param in self.model.parameters():
            param.requires_grad = False

    def _setup_lora(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("Install peft: `pip install peft`") from exc

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        )

        for param in self.model.parameters():
            param.requires_grad = True

        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA initialized (rank=%s)", self.lora_rank)

    def _cache_set(self, sequence: str, embedding: torch.Tensor) -> None:
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return
        self._cache[sequence] = embedding
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _resolve_layer(self, extract_layer: int) -> int:
        if extract_layer < 0:
            return self.model.num_layers
        return extract_layer

    def validate_sequence(self, seq: str, max_length: Optional[int] = None) -> str:
        seq_clean = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYXBZ")
        seq_clean = "".join([c for c in seq_clean if c in valid_aa])
        if len(seq_clean) < 5:
            raise ValueError(f"Sequence too short: {len(seq_clean)} AA")

        model_max = self.MAX_SEQ_LENGTH.get(self.model_name, 1024)
        enforce_max = max_length or model_max
        if len(seq_clean) > enforce_max:
            logger.warning(
                "Sequence %d > limit %d; truncating for model input.",
                len(seq_clean),
                enforce_max,
            )
            seq_clean = seq_clean[:enforce_max]
        return seq_clean

    def embed_sequence(
        self,
        seq: str,
        extract_layer: int = -1,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        seq = self.validate_sequence(seq)
        if seq in self._cache:
            embedding = self._cache[seq]
            return (embedding, None) if return_attention else embedding

        layer = self._resolve_layer(extract_layer)
        data = [("seq", seq)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[layer], return_contacts=return_attention)
        reps = results["representations"][layer]
        embedding = reps[0, 1 : len(seq) + 1].detach()
        self._cache_set(seq, embedding)
        if return_attention:
            contacts = results.get("contacts")
            if contacts is not None:
                contacts = contacts.squeeze(0)
            return embedding, contacts
        return embedding

    def embed_batch(
        self,
        sequences: List[str],
        batch_size: int = 16,
        return_lengths: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], np.ndarray]]:
        all_embeddings: List[torch.Tensor] = []
        all_lengths: List[int] = []
        if not sequences:
            return ([], np.array([], dtype=int)) if return_lengths else []

        logger.info("Embedding %d sequences (batch_size=%d)", len(sequences), batch_size)

        for i in range(0, len(sequences), batch_size):
            batch_raw = sequences[i : i + batch_size]
            batch_clean: List[str] = []
            for seq in batch_raw:
                seq_clean = self.validate_sequence(seq)
                batch_clean.append(seq_clean)
                all_lengths.append(len(seq_clean))

            data = [(f"seq_{i+j}", seq) for j, seq in enumerate(batch_clean)]
            _, _, tokens = self.batch_converter(data)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                results = self.model(tokens, repr_layers=[self.model.num_layers])
            reps = results["representations"][self.model.num_layers]
            for j, seq in enumerate(batch_clean):
                embedding = reps[j, 1 : len(seq) + 1].detach()
                all_embeddings.append(embedding)

        if return_lengths:
            return all_embeddings, np.array(all_lengths, dtype=int)
        return all_embeddings

    def embed_with_structure_guidance(
        self,
        seq: str,
        plddt_scores: np.ndarray,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        embeddings = self.embed_sequence(seq)
        if plddt_scores.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"pLDDT shape {plddt_scores.shape} != sequence length {embeddings.shape[0]}"
            )

        plddt_norm = np.clip(plddt_scores, 0, 100) / 100.0
        weights = torch.tensor(plddt_norm, dtype=embeddings.dtype, device=embeddings.device)
        weights = weights ** (1.0 / temperature)
        weights = weights / (weights.sum() + 1e-10)
        return embeddings * weights.unsqueeze(-1)

    def get_model_info(self) -> Dict[str, Optional[Union[str, int]]]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embed_dim,
            "max_sequence_length": self.MAX_SEQ_LENGTH.get(self.model_name),
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank if self.use_lora else None,
        }
