"""
Pre-flight sanity checks for the truth-bot eval pipeline.

Run before any API calls to catch configuration errors early.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    passed: bool
    errors: list[str] = field(default_factory=list)    # fatal: abort the run
    warnings: list[str] = field(default_factory=list)  # non-fatal: continue but notify


class PreflightChecker:
    """
    Collects errors and warnings from a set of pre-flight checks.

    Each check method appends to self._errors / self._warnings.
    Call run_all() to execute all applicable checks and get a PreflightResult.
    """

    def __init__(self) -> None:
        self._errors: list[str] = []
        self._warnings: list[str] = []

    # ── Individual checks ──────────────────────────────────────────────────────

    def check_api_keys(self, provider: str) -> None:
        """Verify the required API key env var is set and non-empty."""
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        var = key_map.get(provider.lower())
        if var is None:
            self._errors.append(
                f"Unknown provider '{provider}'; expected 'anthropic' or 'openai'"
            )
            return
        val = os.environ.get(var, "")
        if not val:
            self._errors.append(
                f"{var} is not set or empty. "
                f"Set it in your environment or in a .env file at the project root."
            )

    def check_transcript(self, path: "str | Path") -> None:
        """Verify transcript file exists and has enough content."""
        p = Path(path)
        if not p.exists():
            self._errors.append(f"Transcript file not found: {p}")
            return
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) <= 500:
            self._errors.append(
                f"Transcript is too short ({len(text)} chars); must be > 500 chars."
            )
            return
        if len(text) < 5000:
            self._warnings.append(
                f"Transcript is only {len(text)} chars; this may be too short for "
                f"a full speech (expected >= 5000 chars for meaningful results)."
            )

    def check_reference(self, path: "str | Path") -> None:
        """Verify reference JSON is valid, is a list, and has required keys."""
        p = Path(path)
        if not p.exists():
            self._errors.append(f"Reference file not found: {p}")
            return
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            self._errors.append(f"Reference file is not valid JSON: {p} -- {e}")
            return
        if not isinstance(data, list):
            self._errors.append(
                f"Reference file must be a JSON array, got {type(data).__name__}: {p}"
            )
            return
        if len(data) < 3:
            self._errors.append(
                f"Reference file has only {len(data)} items; must have at least 3."
            )
            return
        required_keys = {"id", "claim", "verdict"}
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self._errors.append(f"Reference item {i} is not a dict.")
                return
            missing = required_keys - item.keys()
            if missing:
                self._errors.append(
                    f"Reference item {i} is missing required keys: {sorted(missing)}"
                )
                return
        if len(data) < 10:
            self._warnings.append(
                f"Reference has only {len(data)} claims; >= 10 claims are recommended "
                f"for statistically meaningful fitness evaluation."
            )

    def check_model_not_deprecated(self, model: str) -> None:
        """Warn if the model is in the known-deprecated list."""
        deprecated = [
            "claude-3-5-haiku-20241022",
            "claude-3-haiku-20240307",
            "claude-sonnet-4-5",
            "claude-opus-4-7",
        ]
        if model in deprecated:
            self._warnings.append(
                f"Model '{model}' is deprecated or may produce suboptimal results. "
                f"Consider using a current model such as 'claude-opus-4-9' or 'claude-sonnet-4-9'."
            )

    def check_budget(self, budget_usd: float, dry_run: bool) -> None:
        """Validate the budget parameter for a live run."""
        if dry_run:
            self._warnings.append(
                "DRY-RUN: evolution results will not be meaningful. "
                "All genomes receive identical stub claims -- no selection pressure exists."
            )
            return
        if budget_usd <= 0:
            self._errors.append(
                f"Budget must be > 0 for a live run; got ${budget_usd:.2f}. "
                f"Use --budget to set a limit, or --dry-run to skip API calls."
            )
        elif budget_usd < 0.50:
            self._warnings.append(
                f"Budget is very low (${budget_usd:.2f}); a real run is unlikely to complete. "
                f"Consider at least $1.00 for extraction-only or $5.00 for full synthesis."
            )

    def check_gene_pool_consistency(self) -> None:
        """Verify _GENE_POOL_SIZES in ga.py matches actual variant list lengths in genome.py."""
        try:
            from evolver.ga import _GENE_POOL_SIZES
            from evolver.genome import ExtractionGenome, SynthesisGenome
        except ImportError as e:
            self._errors.append(f"Could not import GA modules for consistency check: {e}")
            return

        # Build a combined lookup: "extraction.X" / "synthesis.X" -> actual pool size
        actual: dict[str, int] = {}
        for gene, variants in ExtractionGenome.GENE_POOLS.items():
            actual[f"extraction.{gene}"] = len(variants)
        for gene, variants in SynthesisGenome.GENE_POOLS.items():
            actual[f"synthesis.{gene}"] = len(variants)

        for key, recorded_size in _GENE_POOL_SIZES.items():
            real_size = actual.get(key)
            if real_size is None:
                self._errors.append(
                    f"_GENE_POOL_SIZES has entry '{key}' that does not match any gene in "
                    f"ExtractionGenome or SynthesisGenome."
                )
            elif real_size != recorded_size:
                self._errors.append(
                    f"_GENE_POOL_SIZES['{key}'] = {recorded_size} but actual variant list "
                    f"has {real_size} entries. The pool size dict is stale -- update ga.py."
                )

    def check_dir_writable(self, path: "str | Path", label: str) -> None:
        """Verify a directory can be created and written to."""
        p = Path(path)
        try:
            p.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=p, prefix=".preflight_", delete=True):
                pass
        except Exception as e:
            self._errors.append(f"{label} directory '{p}' is not writable: {e}")

    # ── Aggregate runner ───────────────────────────────────────────────────────

    def run_all(
        self,
        provider: str = "anthropic",
        transcript_path: "Optional[str | Path]" = None,
        reference_path: "Optional[str | Path]" = None,
        budget_usd: float = 10.0,
        dry_run: bool = False,
        model: str = "",
        results_dir: "Optional[str | Path]" = None,
    ) -> PreflightResult:
        """
        Run all applicable checks and return a PreflightResult.

        Skips individual checks when the relevant parameter is None.
        """
        self._errors = []
        self._warnings = []

        self.check_api_keys(provider)

        if transcript_path is not None:
            self.check_transcript(transcript_path)

        if reference_path is not None:
            self.check_reference(reference_path)

        if model:
            self.check_model_not_deprecated(model)

        # Always check budget (dry_run warning or live-run validation)
        self.check_budget(budget_usd, dry_run)

        self.check_gene_pool_consistency()

        if results_dir is not None:
            self.check_dir_writable(results_dir, "results")

        passed = len(self._errors) == 0
        return PreflightResult(
            passed=passed,
            errors=list(self._errors),
            warnings=list(self._warnings),
        )


def run_preflight(config_dict: dict) -> PreflightResult:
    """
    Convenience wrapper: run all checks from a config dict.

    Expected keys (all optional):
        provider, transcript_path, reference_path, budget_usd,
        dry_run, model, results_dir
    """
    checker = PreflightChecker()
    return checker.run_all(
        provider=config_dict.get("provider", "anthropic"),
        transcript_path=config_dict.get("transcript_path"),
        reference_path=config_dict.get("reference_path"),
        budget_usd=config_dict.get("budget_usd", 10.0),
        dry_run=config_dict.get("dry_run", False),
        model=config_dict.get("model", ""),
        results_dir=config_dict.get("results_dir"),
    )
