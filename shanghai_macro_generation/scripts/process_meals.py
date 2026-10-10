"""Estimate per-meal macros for free-text meal descriptions.

Example:
    python scripts/process_meals.py --input data/raw/shanghai_meals.csv --limit 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings  # noqa: E402
from src.macro_calculator import CONFIDENCE_STRATEGIES  # noqa: E402
from src.parse_meals import PARSER_LLM, PARSER_RULE  # noqa: E402
from src.pipeline import PipelineOptions, format_summary, run_pipeline  # noqa: E402


def setup_logging(level: str, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "process_meals.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(level.upper())
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw" / "shanghai_meals.csv",
                   help="CSV with a meal id column and a meal text column")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "meals_with_macros.csv",
                   help="per-meal output CSV")
    p.add_argument("--breakdown-output", type=Path, default=None,
                   help="per-food output CSV (default: food_breakdown.csv next to --output)")
    p.add_argument("--batch-size", type=int, default=None, help="meals per LLM parse request (default 25)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N meals (for testing)")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="reuse cached parses and nutrition (default). --no-resume recomputes everything "
                        "(manual overrides are never touched)")
    p.add_argument("--force-reparse", action="store_true", help="ignore cached meal parses")
    p.add_argument("--refresh-nutrition", action="store_true",
                   help="ignore cached nutrition results (manual overrides still win)")
    p.add_argument("--skip-llm-fallback", action="store_true",
                   help="do not ask the LLM for foods the database cannot match; they stay unresolved")
    p.add_argument("--parser", choices=[PARSER_LLM, PARSER_RULE], default=PARSER_LLM,
                   help="'llm' (default) or 'rule' (deterministic regex parser, no API calls)")
    p.add_argument("--id-column", default=None, help="input column with meal ids (default meal_id)")
    p.add_argument("--text-column", default=None, help="input column with meal text (default meal_text)")
    p.add_argument("--confidence-strategy", choices=sorted(CONFIDENCE_STRATEGIES), default=None,
                   help="meal confidence formula (default weight_weighted)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    log_file = setup_logging(args.log_level, settings.log_dir)
    logger = logging.getLogger("process_meals")
    logger.info("Logging to %s", log_file)

    options = PipelineOptions(
        input_path=args.input,
        output_path=args.output,
        breakdown_path=args.breakdown_output,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=args.resume,
        force_reparse=args.force_reparse,
        refresh_nutrition=args.refresh_nutrition,
        skip_llm_fallback=args.skip_llm_fallback,
        id_column=args.id_column,
        text_column=args.text_column,
        confidence_strategy=args.confidence_strategy,
        parser=args.parser,
    )
    try:
        summary = run_pipeline(options, settings)
    except KeyboardInterrupt:
        logger.warning("Interrupted. Completed work is checkpointed in %s; rerun to resume.", settings.cache_dir)
        return 130
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    report = format_summary(summary)
    logger.debug("Run summary:\n%s", report)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
