"""Command-line demo for the complete forecasting workflow."""

from __future__ import annotations

import argparse

from .data import generate_matches
from .features import build_features
from .model import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Total Football Forecast demo")
    parser.add_argument("--matches", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    matches = generate_matches(match_count=args.matches, seed=args.seed)
    features = build_features(matches)
    reports = evaluate_models(features)
    print(f"Generated {len(matches)} matches across {matches['home_team'].nunique()} teams\n")
    print(f"{'model':<24} {'accuracy':>9} {'log_loss':>11} {'brier':>9}")
    for report in reports:
        print(
            f"{report.name:<24} {report.accuracy:>9.3f} "
            f"{report.log_loss:>11.3f} {report.brier:>9.3f}"
        )


if __name__ == "__main__":
    main()
