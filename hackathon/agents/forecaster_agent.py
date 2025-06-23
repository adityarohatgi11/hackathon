from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from forecasting.forecaster import Forecaster

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ForecasterAgent(BaseAgent):
    """Agent that listens for engineered feature vectors and outputs forecasts."""

    subscribe_topics = ["feature-vector"]
    publish_topic = "forecast"

    def __init__(self):
        super().__init__(name="ForecasterAgent")
        self._forecaster = Forecaster()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any] | None:
        """Handle incoming feature data and return forecast."""
        try:
            # Expect message to have a list of historical prices under key 'prices'
            prices_raw = message.get("prices")
            if not prices_raw:
                logger.warning("No prices in message – skipping")
                return None

            prices_df = pd.DataFrame(prices_raw)
            prices_df["timestamp"] = pd.to_datetime(prices_df["timestamp"])

            # Generate forecast for next 24 hours
            prediction_df = self._forecaster.predict_next(prices_df, periods=24)

            # Convert to records for JSON serialisation
            return {
                "forecast": prediction_df.to_dict(orient="records"),
                "source": self.name,
            }
        except Exception as exc:  # pragma: no cover
            logger.exception("Forecasting failed: %s", exc)
            return None


if __name__ == "__main__":
    ForecasterAgent().start() 