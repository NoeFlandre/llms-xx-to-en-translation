"""Model calls for COMET-22, MetricX-24 and GlotLID, each loaded once and reused for every shard.

Runs in the metrics environment on Grid5000 nodes (``scripts/g5k/setup_metrics.sh``). Imports are
deferred so the package imports without those dependencies installed.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xx_to_en.scoring import glotlid_code, metricx_text, pad_sequences

COMET_MODEL = "Unbabel/wmt22-comet-da"
METRICX_MODEL = "google/metricx-24-hybrid-xl-v2p6"
METRICX_TOKENIZER = "google/mt5-xl"
GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model.bin"


def _load_metricx(repo: Path) -> tuple[Any, Any, Any]:
    """Tokenizer, model and device for MetricX, using the upstream model class from ``repo``."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import torch  # ty: ignore[import-not-found]
    import transformers  # ty: ignore[import-not-found]
    from metricx24 import models  # ty: ignore[import-not-found]

    # Slow (SentencePiece) tokenizer: the fast conversion imports protobuf generated code that
    # unbabel-comet's protobuf<5 pin rules out. The sentencepiece model is the same either way.
    tokenizer = transformers.AutoTokenizer.from_pretrained(METRICX_TOKENIZER, use_fast=False)
    device = torch.device("cuda")
    model = models.MT5ForRegression.from_pretrained(METRICX_MODEL, torch_dtype="auto")
    model.to(device)
    model.eval()
    return tokenizer, model, device


class Scorers:
    """Lazy, load-once metric models shared by every shard of a job."""

    def __init__(
        self,
        *,
        metricx_repo: Path,
        work: Path,
        comet_batch: int = 64,
        metricx_batch: int = 32,
        max_input_length: int = 1536,
    ) -> None:
        self._metricx_repo = metricx_repo
        self._work = work
        self._comet_batch = comet_batch
        self._metricx_batch = metricx_batch
        self._max_input_length = max_input_length
        self._comet: Any = None
        self._metricx: tuple[Any, Any, Any] | None = None
        self._lid: Any = None

    def comet(self, inputs: Sequence[dict[str, str]]) -> list[float]:
        """COMET-22 reference-based scores, one per input, in order."""
        if not inputs:
            return []
        if self._comet is None:
            from comet import download_model, load_from_checkpoint  # ty: ignore[import-not-found]

            self._comet = load_from_checkpoint(download_model(COMET_MODEL))
        output = self._comet.predict(list(inputs), batch_size=self._comet_batch, gpus=1, progress_bar=False)
        return [float(score) for score in output.scores]

    def metricx(self, inputs: Sequence[dict[str, str]]) -> list[float]:
        """MetricX-24 hybrid XL, reference-based: tokenize as upstream does, batch by length, regress."""
        if not inputs:
            return []
        import torch  # ty: ignore[import-not-found]

        if self._metricx is None:
            self._metricx = _load_metricx(self._metricx_repo)
        tokenizer, model, device = self._metricx
        rows = [
            tokenizer(metricx_text(row), max_length=self._max_input_length, truncation=True, padding=False)[
                "input_ids"
            ][:-1]
            for row in inputs
        ]
        order = sorted(range(len(rows)), key=lambda index: len(rows[index]))
        scores = [0.0] * len(rows)
        with torch.no_grad():
            for start in range(0, len(order), self._metricx_batch):
                chunk = order[start : start + self._metricx_batch]
                ids, masks = pad_sequences([rows[index] for index in chunk], tokenizer.pad_token_id)
                output = model(
                    input_ids=torch.tensor(ids, device=device),
                    attention_mask=torch.tensor(masks, device=device),
                    use_cache=False,  # the decoder cache breaks this upstream model on transformers 4.57
                )
                for index, value in zip(chunk, output.predictions.float().cpu().tolist(), strict=True):
                    scores[index] = float(value)
        return scores

    def glotlid(self, texts: Sequence[str]) -> list[str]:
        """Detected language per text as corpus-style codes (``eng_Latn``). Empty texts give ``""``."""
        if self._lid is None:
            import fasttext  # ty: ignore[import-not-found]
            from huggingface_hub import hf_hub_download

            self._lid = fasttext.load_model(hf_hub_download(repo_id=GLOTLID_REPO, filename=GLOTLID_FILE))
        codes = []
        for text in texts:
            flat = text.replace("\n", " ").strip()
            if not flat:
                codes.append("")
                continue
            labels, _ = self._lid.predict(flat)
            codes.append(glotlid_code(labels[0]))
        return codes
