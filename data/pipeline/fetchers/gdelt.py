"""
GDELT Fetcher - Global Database of Events, Language, and Tone

Fetches global event data from GDELT.
Documentation: https://www.gdeltproject.org/

Note: Full GDELT access requires Google BigQuery. This fetcher uses
the GDELT Analysis Service for summary statistics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

from .base import BaseFetcher, FetchError, RateLimitConfig

logger = logging.getLogger(__name__)


class GDELTFetcher(BaseFetcher):
    """
    Fetcher for GDELT event data.

    Uses GDELT Analysis Service API for summary data.
    For full dataset access, use BigQuery directly.

    Example:
        >>> fetcher = GDELTFetcher()
        >>> result = fetcher.fetch(
        ...     "timeline",
        ...     start_date="2020-01-01",
        ...     query="climate change",
        ... )
    """

    # GDELT Analysis Service endpoints
    DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
    GEO_API = "https://api.gdeltproject.org/api/v2/geo/geo"
    TV_API = "https://api.gdeltproject.org/api/v2/tv/tv"

    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=60,
        min_delay_seconds=1.0,
    )

    def __init__(
        self,
        cache: Optional[Any] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        if requests is None:
            raise ImportError("requests library required")

        super().__init__(
            api_key=None,
            cache=cache,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        return "gdelt"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch GDELT data.

        Args:
            series_id: Query type ('timeline', 'tonechart', 'wordcloud')
            start_date: Start date
            end_date: End date
            **kwargs:
                - query: Search query
                - mode: 'artlist', 'timelinevol', 'timelinetone', etc.
                - format: 'json', 'csv'
        """
        query = kwargs.get('query', '')
        mode = kwargs.get('mode', 'timelinevol')

        params = {
            'query': query,
            'mode': mode,
            'format': 'json',
        }

        if start_date:
            params['startdatetime'] = start_date.strftime('%Y%m%d%H%M%S')
        if end_date:
            params['enddatetime'] = end_date.strftime('%Y%m%d%H%M%S')

        try:
            response = self._session.get(self.DOC_API, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"GDELT request failed: {e}")

        # Parse timeline response
        timeline = data.get('timeline', [])
        if not timeline:
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        rows = []
        for point in timeline:
            rows.append({
                'date': pd.to_datetime(point.get('date', '')),
                'value': point.get('value', 0),
                'series_id': f"gdelt_{mode}_{query[:20]}",
            })

        return pd.DataFrame(rows)

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available GDELT query modes."""
        modes = [
            {'id': 'timelinevol', 'name': 'Article Volume Timeline'},
            {'id': 'timelinetone', 'name': 'Tone Timeline'},
            {'id': 'tonechart', 'name': 'Tone Distribution'},
            {'id': 'wordcloud', 'name': 'Word Cloud'},
        ]
        return modes


__all__ = ['GDELTFetcher']
