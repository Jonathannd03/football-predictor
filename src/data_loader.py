import pandas as pd
import os

# ─────────────────────────────────────────────
# League codes (football-data.co.uk)
# ─────────────────────────────────────────────
LEAGUES = {
    "bundesliga":       "D1",
    "premier_league":   "E0",
    "la_liga":          "SP1",
    "serie_a":          "I1",
    "ligue_1":          "F1",
}

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_league_data(league: str, season: str) -> pd.DataFrame:
    """
    Fetch a single season from football-data.co.uk.

    Args:
        league:  League code e.g. 'D1', 'E0'  (or friendly name e.g. 'bundesliga')
        season:  Season string e.g. '2324' for 2023/24

    Returns:
        DataFrame with raw match data
    """
    code = LEAGUES.get(league, league)  # accept both friendly names and raw codes
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
    print(f"  → Fetching {code} {season} from {url}")
    df = pd.read_csv(url)
    df["season"] = season
    df["league"] = code
    return df


def load_multiple_seasons(league: str, seasons: list[str]) -> pd.DataFrame:
    """
    Load and concatenate multiple seasons for a given league.

    Args:
        league:   League name or code
        seasons:  List of season strings e.g. ['2324', '2223', '2122']

    Returns:
        Combined DataFrame sorted by date
    """
    frames = []
    for s in seasons:
        try:
            df = load_league_data(league, s)
            frames.append(df)
        except Exception as e:
            print(f"  ⚠️  Could not load {league} {s}: {e}")

    if not frames:
        raise ValueError(f"No data loaded for {league}")

    combined = pd.concat(frames, ignore_index=True)

    # Parse date
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined = combined.sort_values("Date").reset_index(drop=True)

    print(f"\n✅ Loaded {len(combined)} matches across {len(seasons)} seasons")
    return combined


def save_raw(df: pd.DataFrame, filename: str) -> None:
    """Save raw DataFrame to data/raw/"""
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved to {path}")


def load_raw(filename: str) -> pd.DataFrame:
    """Load a previously saved raw CSV."""
    path = os.path.join(RAW_DIR, filename)
    return pd.read_csv(path, parse_dates=["Date"])


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    seasons = ["2324", "2223", "2122", "2021", "1920"]
    df = load_multiple_seasons("bundesliga", seasons)
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].head(10))
    save_raw(df, "bundesliga_raw.csv")
