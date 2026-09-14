import pandas as pd
import pytest

from src.sentiment import aggregate_daily, score_headlines


@pytest.fixture
def headlines():
    return pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "Headline": [
            "Company reports record profits and strong growth",
            "Company faces lawsuit over safety failures",
            "Analysts upgrade stock to buy",
        ],
        "Stock Name": ["TEST", "TEST", "TEST"],
    })


class TestScoring:
    def test_polarity_column_added(self, headlines):
        assert "polarity_score" in score_headlines(headlines).columns

    def test_positive_and_negative_are_separated(self, headlines):
        scored = score_headlines(headlines)
        assert scored.loc[0, "polarity_score"] > 0
        assert scored.loc[1, "polarity_score"] < 0

    def test_scores_are_bounded(self, headlines):
        scored = score_headlines(headlines)
        assert scored["polarity_score"].between(-1, 1).all()

    def test_score_is_continuous_not_bucketed(self, headlines):
        # Magnitude must survive — bucketing to -1/0/1 would lose the
        # distinction the trading signal depends on.
        scored = score_headlines(headlines)
        assert not scored["polarity_score"].isin([-1, 0, 1]).all()

    def test_does_not_mutate_input(self, headlines):
        score_headlines(headlines)
        assert "polarity_score" not in headlines.columns


class TestAggregation:
    def test_collapses_to_one_row_per_day(self, headlines):
        result = aggregate_daily(score_headlines(headlines))
        assert len(result) == 2

    def test_uses_mean_not_sum(self, headlines):
        # A busy news day must not get proportionally more signal strength.
        scored = score_headlines(headlines)
        result = aggregate_daily(scored)
        day1 = result[result["Date"] == "2024-01-01"]["polarity_score"].iloc[0]
        expected = scored.iloc[:2]["polarity_score"].mean()
        assert day1 == pytest.approx(expected)

    def test_headline_count_retained(self, headlines):
        result = aggregate_daily(score_headlines(headlines))
        assert result[result["Date"] == "2024-01-01"]["headline_count"].iloc[0] == 2
