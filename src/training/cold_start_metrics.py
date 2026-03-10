"""Unit 1: Cold-start evaluation harness — regime tracking and summary stats."""
import pandas as pd


def classify_regime(n_ctx: int) -> str:
    """Classify episode into cold / sparse / warm based on context size."""
    if n_ctx == 0:
        return "cold"
    elif n_ctx < 8:
        return "sparse"
    else:
        return "warm"


def summarize_cold_start(episode_log: list) -> pd.DataFrame:
    """
    Given the episode log (list of dicts with 'regime', 'ci', 'ef10'),
    return a summary DataFrame with mean/std/count of CI and EF10 per regime.
    """
    df = pd.DataFrame(episode_log)
    if "regime" not in df.columns or df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby("regime")[["ci", "ef10"]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    return summary
