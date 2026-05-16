"""
UN COMTRADE Fetcher

Fetches international trade data from UN COMTRADE API.
API Documentation: https://comtradeapi.un.org/

Requires API key from: https://comtradeplus.un.org/
"""

from __future__ import annotations

import os
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


class COMTRADEFetcher(BaseFetcher):
    """
    Fetcher for UN COMTRADE trade data.

    Example:
        >>> fetcher = COMTRADEFetcher(api_key="your_key")
        >>> result = fetcher.fetch(
        ...     "HS",  # Harmonized System
        ...     start_date="2020-01-01",
        ...     reporter="USA",
        ...     partner="CHN",
        ...     commodity_code="8471",  # Computers
        ... )
    """

    BASE_URL = "https://comtradeapi.un.org/data/v1/get"

    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=10,  # Free tier: 100/hour
        min_delay_seconds=6.0,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        if requests is None:
            raise ImportError("requests library required")

        api_key = api_key or os.environ.get('COMTRADE_API_KEY')
        if not api_key:
            raise ValueError(
                "COMTRADE API key required. Set COMTRADE_API_KEY environment variable "
                "or register at https://comtradeplus.un.org/"
            )

        super().__init__(
            api_key=api_key,
            cache=cache,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )
        self._session = requests.Session()
        self._session.headers.update({
            'Ocp-Apim-Subscription-Key': api_key,
        })

    @property
    def source_name(self) -> str:
        return "comtrade"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch COMTRADE data.

        Args:
            series_id: Classification ('HS', 'SITC', 'BEC')
            start_date: Start period
            end_date: End period
            **kwargs:
                - reporter: Reporter country code
                - partner: Partner country code
                - commodity_code: Commodity code
                - flow: 'M' (import), 'X' (export)
        """
        params = {
            'typeCode': series_id,
            'freqCode': 'A',  # Annual
            'flowCode': kwargs.get('flow', 'M,X'),
            'reporterCode': kwargs.get('reporter', ''),
            'partnerCode': kwargs.get('partner', ''),
            'cmdCode': kwargs.get('commodity_code', 'TOTAL'),
        }

        if start_date:
            params['period'] = start_date.year
        if end_date and start_date:
            # COMTRADE expects period range as comma-separated years
            years = list(range(start_date.year, end_date.year + 1))
            params['period'] = ','.join(str(y) for y in years)

        # Remove empty params
        params = {k: v for k, v in params.items() if v}

        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"COMTRADE request failed: {e}")

        records = data.get('data', [])
        if not records:
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        rows = []
        for rec in records:
            rows.append({
                'date': pd.to_datetime(f"{rec.get('period', 2000)}-01-01"),
                'value': rec.get('primaryValue'),
                'series_id': f"comtrade_{rec.get('cmdCode', '')}",
                'reporter': rec.get('reporterDesc'),
                'partner': rec.get('partnerDesc'),
                'commodity': rec.get('cmdDesc'),
                'flow': rec.get('flowDesc'),
                'quantity': rec.get('qty'),
                'unit': rec.get('qtyUnitAbbr'),
            })

        return pd.DataFrame(rows)

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available classifications."""
        return [
            {'id': 'HS', 'name': 'Harmonized System'},
            {'id': 'SITC', 'name': 'Standard International Trade Classification'},
            {'id': 'BEC', 'name': 'Broad Economic Categories'},
        ]


__all__ = ['COMTRADEFetcher']
