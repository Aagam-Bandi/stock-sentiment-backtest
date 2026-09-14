"""Sentiment scoring for news headlines.

Uses VADER, a rule-based lexicon scorer tuned for short informal text. It is
used directly rather than as a labelling step for a downstream classifier —
see README "What went wrong" for why the original supervised setup was
circular.
"""

import logging
import unicodedata

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


def ensure_lexicon() -> None:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        logger.info("downloading VADER lexicon")
        nltk.download("vader_lexicon", quiet=True)


def score_headlines(df: pd.DataFrame, column: str = "Headline") -> pd.DataFrame:
    """Attach VADER compound polarity to each headline.

    The compound score is retained as a continuous value rather than bucketed
    into positive/neutral/negative. Bucketing throws away magnitude, and
    magnitude is what distinguishes a mildly positive headline from a strongly
    positive one — which is exactly the distinction a trading signal needs.
    """
    ensure_lexicon()
    analyzer = SentimentIntensityAnalyzer()

    df = df.copy()
    df["polarity_score"] = df[column].apply(
        lambda text: analyzer.polarity_scores(
            unicodedata.normalize("NFKD", str(text))
        )["compound"]
    )
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple same-day headlines per ticker into one score.

    Mean rather than sum: a ticker with twenty headlines on a busy news day
    should not get twenty times the signal strength of one with a single
    headline.
    """
    return (
        df.groupby(["Date", "Stock Name"], as_index=False)
        .agg(polarity_score=("polarity_score", "mean"),
             headline_count=("polarity_score", "size"))
    )
