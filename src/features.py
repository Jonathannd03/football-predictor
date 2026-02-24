"""
features.py — Feature engineering for football match prediction

For each match, we compute features based on:
  - Recent form (last N matches, home/away split)
  - Rolling goal averages (scored & conceded)
  - Head-to-head history
  - Rest days (fatigue)
  - Season context (match number, league position proxy)

All features are computed using only data BEFORE the match date
to avoid data leakage.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _result_points(ftr: str, perspective: str) -> float:
    """Convert FTR (H/D/A) to points from a team's perspective."""
    if perspective == "home":
        return {"H": 3.0, "D": 1.0, "A": 0.0}.get(ftr, np.nan)
    else:
        return {"A": 3.0, "D": 1.0, "H": 0.0}.get(ftr, np.nan)


def _result_win(ftr: str, perspective: str) -> float:
    """1 if win, 0 if draw/loss."""
    if perspective == "home":
        return 1.0 if ftr == "H" else 0.0
    else:
        return 1.0 if ftr == "A" else 0.0


# ─────────────────────────────────────────────────────────────
# FORM FEATURES
# ─────────────────────────────────────────────────────────────

def compute_form(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    For each match, compute the rolling form of home and away teams
    over the last N matches (across home and away games combined).

    Features added:
      - home_form_pts      — avg points per game (last N)
      - away_form_pts
      - home_form_wins     — win rate (last N)
      - away_form_wins
      - home_goals_scored  — avg goals scored (last N)
      - away_goals_scored
      - home_goals_conceded
      - away_goals_conceded
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Build a per-team match history (perspective-normalised)
    records = []
    for _, row in df.iterrows():
        records.append({
            "date":     row["Date"],
            "team":     row["HomeTeam"],
            "opponent": row["AwayTeam"],
            "match_idx": _,
            "goals_scored":    row["FTHG"],
            "goals_conceded":  row["FTAG"],
            "points":   _result_points(row["FTR"], "home"),
            "win":      _result_win(row["FTR"], "home"),
        })
        records.append({
            "date":     row["Date"],
            "team":     row["AwayTeam"],
            "opponent": row["HomeTeam"],
            "match_idx": _,
            "goals_scored":    row["FTAG"],
            "goals_conceded":  row["FTHG"],
            "points":   _result_points(row["FTR"], "away"),
            "win":      _result_win(row["FTR"], "away"),
        })

    history = pd.DataFrame(records).sort_values("date")

    def get_form(team: str, before_date: pd.Timestamp, n: int):
        """Get last N matches for a team before a given date."""
        past = history[(history["team"] == team) & (history["date"] < before_date)]
        last_n = past.tail(n)
        if len(last_n) == 0:
            return {
                "form_pts": np.nan,
                "form_wins": np.nan,
                "goals_scored": np.nan,
                "goals_conceded": np.nan,
                "matches_played": 0,
            }
        return {
            "form_pts":       last_n["points"].mean(),
            "form_wins":      last_n["win"].mean(),
            "goals_scored":   last_n["goals_scored"].mean(),
            "goals_conceded": last_n["goals_conceded"].mean(),
            "matches_played": len(last_n),
        }

    # Compute for each match
    home_forms, away_forms = [], []
    for _, row in df.iterrows():
        home_forms.append(get_form(row["HomeTeam"], row["Date"], n))
        away_forms.append(get_form(row["AwayTeam"], row["Date"], n))

    home_df = pd.DataFrame(home_forms).add_prefix("home_")
    away_df = pd.DataFrame(away_forms).add_prefix("away_")

    df = pd.concat([df.reset_index(drop=True), home_df, away_df], axis=1)
    return df


# ─────────────────────────────────────────────────────────────
# HEAD-TO-HEAD FEATURES
# ─────────────────────────────────────────────────────────────

def compute_h2h(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    For each match, compute head-to-head stats between the two teams
    over the last N encounters.

    Features added:
      - h2h_home_win_rate   — home team's win rate in past H2H
      - h2h_draw_rate
      - h2h_away_win_rate
      - h2h_home_goals_avg
      - h2h_away_goals_avg
      - h2h_matches         — how many H2H matches found
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    h2h_features = []

    for i, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        date = row["Date"]

        # Find past meetings (both directions)
        past = df[
            (df["Date"] < date) &
            (
                ((df["HomeTeam"] == home) & (df["AwayTeam"] == away)) |
                ((df["HomeTeam"] == away) & (df["AwayTeam"] == home))
            )
        ].tail(n)

        if len(past) == 0:
            h2h_features.append({
                "h2h_home_win_rate":  np.nan,
                "h2h_draw_rate":      np.nan,
                "h2h_away_win_rate":  np.nan,
                "h2h_home_goals_avg": np.nan,
                "h2h_away_goals_avg": np.nan,
                "h2h_matches":        0,
            })
            continue

        # Normalise results from home team's perspective
        home_wins, draws, away_wins = 0, 0, 0
        home_goals, away_goals = [], []

        for _, r in past.iterrows():
            if r["HomeTeam"] == home:
                # Standard direction
                home_goals.append(r["FTHG"])
                away_goals.append(r["FTAG"])
                if r["FTR"] == "H": home_wins += 1
                elif r["FTR"] == "D": draws += 1
                else: away_wins += 1
            else:
                # Reversed direction — flip perspective
                home_goals.append(r["FTAG"])
                away_goals.append(r["FTHG"])
                if r["FTR"] == "A": home_wins += 1
                elif r["FTR"] == "D": draws += 1
                else: away_wins += 1

        total = len(past)
        h2h_features.append({
            "h2h_home_win_rate":  home_wins / total,
            "h2h_draw_rate":      draws / total,
            "h2h_away_win_rate":  away_wins / total,
            "h2h_home_goals_avg": np.mean(home_goals),
            "h2h_away_goals_avg": np.mean(away_goals),
            "h2h_matches":        total,
        })

    h2h_df = pd.DataFrame(h2h_features)
    df = pd.concat([df.reset_index(drop=True), h2h_df], axis=1)
    return df


# ─────────────────────────────────────────────────────────────
# REST DAYS FEATURES
# ─────────────────────────────────────────────────────────────

def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days since last match for each team (home & away).

    Features added:
      - home_rest_days
      - away_rest_days
      - rest_advantage    — home_rest_days - away_rest_days
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # Collect all match dates per team
    team_last_match: dict = {}

    home_rest, away_rest = [], []

    for _, row in df.iterrows():
        home, away, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # Rest days for home team
        if home in team_last_match and pd.notna(date):
            home_rest.append((date - team_last_match[home]).days)
        else:
            home_rest.append(np.nan)  # first match — no prior data

        # Rest days for away team
        if away in team_last_match and pd.notna(date):
            away_rest.append((date - team_last_match[away]).days)
        else:
            away_rest.append(np.nan)

        # Update last match date for both teams
        if pd.notna(date):
            team_last_match[home] = date
            team_last_match[away] = date

    df["home_rest_days"] = home_rest
    df["away_rest_days"] = away_rest
    df["rest_advantage"] = df["home_rest_days"] - df["away_rest_days"]

    return df


# ─────────────────────────────────────────────────────────────
# SEASON CONTEXT
# ─────────────────────────────────────────────────────────────

def compute_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add season context: match week number within the season.

    Features added:
      - match_week   — sequential match number in the season (1-34 for Bundesliga)
    """
    df = df.copy().sort_values(["season", "Date"]).reset_index(drop=True)
    df["match_week"] = df.groupby("season").cumcount() // 9 + 1  # ~9 matches per matchday
    return df


# ─────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ─────────────────────────────────────────────────────────────

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the match result as a numeric target.

      FTR → target
        H  → 0  (home win)
        D  → 1  (draw)
        A  → 2  (away win)
    """
    mapping = {"H": 0, "D": 1, "A": 2}
    df = df.copy()
    df["target"] = df["FTR"].map(mapping)
    return df


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, form_n: int = 5, h2h_n: int = 5) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df:      Raw match DataFrame (from data_loader)
        form_n:  Number of recent matches for form calculation
        h2h_n:   Number of H2H matches to look back

    Returns:
        DataFrame with all features + target column
    """
    print("🔧 Building features...")

    print("  [1/5] Computing form features...")
    df = compute_form(df, n=form_n)

    print("  [2/5] Computing H2H features...")
    df = compute_h2h(df, n=h2h_n)

    print("  [3/5] Computing rest days...")
    df = compute_rest_days(df)

    print("  [4/5] Adding season context...")
    df = compute_season_context(df)

    print("  [5/5] Encoding target...")
    df = encode_target(df)

    # Drop rows with no form data (first few matches of the dataset)
    df = df.dropna(subset=["home_form_pts", "away_form_pts"])

    print(f"\n✅ Feature matrix ready: {df.shape[0]} matches × {df.shape[1]} columns")
    return df


def get_feature_columns() -> list[str]:
    """Returns the list of feature column names used for training."""
    return [
        # Form
        "home_form_pts",
        "home_form_wins",
        "home_goals_scored",
        "home_goals_conceded",
        "away_form_pts",
        "away_form_wins",
        "away_goals_scored",
        "away_goals_conceded",
        # H2H
        "h2h_home_win_rate",
        "h2h_draw_rate",
        "h2h_away_win_rate",
        "h2h_home_goals_avg",
        "h2h_away_goals_avg",
        # Rest
        "home_rest_days",
        "away_rest_days",
        "rest_advantage",
        # Context
        "match_week",
    ]


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_multiple_seasons

    seasons = ["2324", "2223", "2122"]
    raw = load_multiple_seasons("bundesliga", seasons)
    features = build_features(raw)

    print("\nFeature preview:")
    print(features[get_feature_columns() + ["target"]].head())
    print("\nTarget distribution:")
    print(features["target"].value_counts(normalize=True).round(3))
