"""Chronological pre-match feature engineering."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TeamState:
    rating: float = 1500.0
    points: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def mean(self, values: deque[float]) -> float:
        return float(np.mean(values)) if values else 0.0


FEATURE_COLUMNS = [
    "home_form",
    "away_form",
    "home_goals_for",
    "away_goals_for",
    "home_goals_against",
    "away_goals_against",
    "rating_difference",
    "home_advantage",
]


def _expected_home(home_rating: float, away_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating - 65.0) / 400.0))


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build features using only results available before each match."""

    required = {
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "result",
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    ordered = matches.sort_values(["date", "match_id"] if "match_id" in matches else ["date"])
    states: defaultdict[str, TeamState] = defaultdict(TeamState)
    feature_rows: list[dict[str, object]] = []

    for row in ordered.itertuples(index=False):
        home = states[str(row.home_team)]
        away = states[str(row.away_team)]
        feature_rows.append(
            {
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_form": home.mean(home.points) / 3.0,
                "away_form": away.mean(away.points) / 3.0,
                "home_goals_for": home.mean(home.goals_for),
                "away_goals_for": away.mean(away.goals_for),
                "home_goals_against": home.mean(home.goals_against),
                "away_goals_against": away.mean(away.goals_against),
                "rating_difference": (home.rating - away.rating) / 400.0,
                "home_advantage": 1.0,
                "result": row.result,
            }
        )

        if row.result == "H":
            home_points, away_points, actual_home = 3.0, 0.0, 1.0
        elif row.result == "A":
            home_points, away_points, actual_home = 0.0, 3.0, 0.0
        else:
            home_points, away_points, actual_home = 1.0, 1.0, 0.5

        expected = _expected_home(home.rating, away.rating)
        change = 24.0 * (actual_home - expected)
        home.rating += change
        away.rating -= change
        home.points.append(home_points)
        away.points.append(away_points)
        home.goals_for.append(float(row.home_goals))
        home.goals_against.append(float(row.away_goals))
        away.goals_for.append(float(row.away_goals))
        away.goals_against.append(float(row.home_goals))

    return pd.DataFrame(feature_rows)
