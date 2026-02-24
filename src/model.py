"""
model.py — Training, evaluation, and value bet detection

Model: XGBoost multi-class classifier (H / D / A)
Output: Probabilities for each outcome
Value: Compare model probability vs bookmaker implied probability
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import os

from features import get_feature_columns

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame, save: bool = True) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier on the feature matrix.

    Uses TimeSeriesSplit to respect temporal order — no future leakage.

    Args:
        df:    Feature DataFrame (output of build_features)
        save:  Save model to disk

    Returns:
        Trained XGBClassifier
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    y = df["target"]

    print(f"📊 Training on {len(X)} matches | {X.shape[1]} features")
    print(f"   Target distribution: {y.value_counts().to_dict()}")

    # Time-series cross-validation (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        cv_scores.append(acc)
        print(f"   Fold {fold+1}: accuracy = {acc:.3f}")

    print(f"\n🎯 Mean CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # Final model on all data
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "xgb_model.joblib")
        joblib.dump(final_model, path)
        print(f"💾 Model saved to {path}")

    return final_model


def load_model() -> xgb.XGBClassifier:
    """Load a saved model from disk."""
    path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_proba(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict outcome probabilities for a set of matches.

    Returns DataFrame with columns:
      prob_home, prob_draw, prob_away
    """
    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)
    probs = model.predict_proba(X)

    return pd.DataFrame({
        "prob_home": probs[:, 0],
        "prob_draw": probs[:, 1],
        "prob_away": probs[:, 2],
    }, index=df.index)


# ─────────────────────────────────────────────────────────────
# VALUE BET DETECTION
# ─────────────────────────────────────────────────────────────

def detect_value_bets(
    df: pd.DataFrame,
    probs: pd.DataFrame,
    min_edge: float = 0.05,
    home_odds_col: str = "B365H",
    draw_odds_col: str = "B365D",
    away_odds_col: str = "B365A",
) -> pd.DataFrame:
    """
    Compare model probabilities vs bookmaker implied probabilities.

    A value bet exists when:
        model_prob > implied_prob + min_edge

    Args:
        df:           Match DataFrame (must contain odds columns)
        probs:        Output of predict_proba()
        min_edge:     Minimum edge required (default 5%)
        *_odds_col:   Column names for bookmaker odds (Bet365 by default)

    Returns:
        DataFrame of value bets with edge and recommended stake
    """
    results = []

    for i in df.index:
        row = df.loc[i]
        p = probs.loc[i]

        # Bookmaker implied probabilities (accounting for overround)
        try:
            h_odds = float(row[home_odds_col])
            d_odds = float(row[draw_odds_col])
            a_odds = float(row[away_odds_col])
        except (KeyError, ValueError):
            continue

        if any(pd.isna([h_odds, d_odds, a_odds])):
            continue

        implied_h = 1 / h_odds
        implied_d = 1 / d_odds
        implied_a = 1 / a_odds

        # Check each outcome for value
        for outcome, model_prob, implied_prob, odds in [
            ("H", p["prob_home"], implied_h, h_odds),
            ("D", p["prob_draw"], implied_d, d_odds),
            ("A", p["prob_away"], implied_a, a_odds),
        ]:
            edge = model_prob - implied_prob
            if edge >= min_edge:
                results.append({
                    "date":         row.get("Date"),
                    "home_team":    row.get("HomeTeam"),
                    "away_team":    row.get("AwayTeam"),
                    "outcome":      outcome,
                    "model_prob":   round(model_prob, 3),
                    "implied_prob": round(implied_prob, 3),
                    "edge":         round(edge, 3),
                    "odds":         odds,
                    "actual":       row.get("FTR"),
                })

    value_df = pd.DataFrame(results)
    if len(value_df) > 0:
        value_df = value_df.sort_values("edge", ascending=False).reset_index(drop=True)

    return value_df


# ─────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────

def backtest(value_bets: pd.DataFrame, stake: float = 1.0) -> dict:
    """
    Simulate betting on all detected value bets with a flat stake.

    Args:
        value_bets:  Output of detect_value_bets() (with 'actual' column)
        stake:       Stake per bet (default £1)

    Returns:
        Dict with P&L, ROI, win rate, and bet count
    """
    if value_bets.empty:
        return {"bets": 0, "roi": 0, "pnl": 0, "win_rate": 0}

    pnl = 0.0
    wins = 0

    for _, row in value_bets.iterrows():
        if row["actual"] == row["outcome"]:
            pnl += stake * (row["odds"] - 1)
            wins += 1
        else:
            pnl -= stake

    total_staked = len(value_bets) * stake

    return {
        "bets":      len(value_bets),
        "wins":      wins,
        "win_rate":  round(wins / len(value_bets), 3),
        "pnl":       round(pnl, 2),
        "roi":       round(pnl / total_staked * 100, 2),
    }


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from data_loader import load_multiple_seasons
    from features import build_features

    print("Loading data...")
    raw = load_multiple_seasons("bundesliga", ["2324", "2223", "2122", "2021"])

    print("\nBuilding features...")
    df = build_features(raw)

    print("\nTraining model...")
    model = train_model(df)

    print("\nPredicting probabilities...")
    probs = predict_proba(model, df)

    print("\nDetecting value bets...")
    value_bets = detect_value_bets(df, probs)
    print(f"Found {len(value_bets)} value bets")

    if not value_bets.empty:
        print("\nTop 10 value bets:")
        print(value_bets.head(10).to_string())

        print("\nBacktest results:")
        results = backtest(value_bets)
        for k, v in results.items():
            print(f"  {k}: {v}")
