"""Deterministic synthetic match data for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeagueConfig:
    teams: int = 20
    home_advantage: float = 0.24
    base_goals: float = 1.28


def generate_matches(
    match_count: int = 600,
    seed: int = 42,
    config: LeagueConfig = LeagueConfig(),
) -> pd.DataFrame:
    """Generate plausible chronological soccer results without external data.

    Latent team strengths drift slowly through time. Goals are sampled from
    Poisson distributions whose rates depend on the teams and home advantage.
    """

    if match_count < config.teams:
        raise ValueError("match_count must be at least the number of teams")

    rng = np.random.default_rng(seed)
    team_names = np.array([f"FC {index + 1:02d}" for index in range(config.teams)])
    strengths = rng.normal(0.0, 0.38, config.teams)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2022-08-01")

    for match_id in range(match_count):
        if match_id and match_id % (config.teams * 2) == 0:
            strengths += rng.normal(0.0, 0.035, config.teams)

        home_idx, away_idx = rng.choice(config.teams, size=2, replace=False)
        home_rate = config.base_goals * np.exp(
            config.home_advantage + strengths[home_idx] - strengths[away_idx]
        )
        away_rate = config.base_goals * np.exp(
            strengths[away_idx] - strengths[home_idx]
        )
        home_goals = int(rng.poisson(np.clip(home_rate, 0.25, 3.8)))
        away_goals = int(rng.poisson(np.clip(away_rate, 0.25, 3.8)))
        result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"

        rows.append(
            {
                "match_id": match_id,
                "date": start + pd.Timedelta(days=match_id // 5),
                "home_team": str(team_names[home_idx]),
                "away_team": str(team_names[away_idx]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": result,
            }
        )

    return pd.DataFrame(rows)
