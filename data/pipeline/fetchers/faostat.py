"""
FAOSTAT Fetcher - UN Food and Agriculture Organization

Fetches agricultural and food security data from FAOSTAT.
API Documentation: https://fenixservices.fao.org/faostat/api/v1/

Key domains for RATCHET:
    - QCL: Crops and livestock products
    - FBS: Food balances
    - RL: Land use
    - OA: Population
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


# FAOSTAT domain codes
FAOSTAT_DOMAINS = {
    'QCL': 'Crops and livestock products',
    'FBS': 'Food Balances (old)',
    'FBSH': 'Food Balances (new)',
    'RL': 'Land Use',
    'OA': 'Population',
    'PP': 'Producer Prices',
    'PI': 'Price Indices',
    'TM': 'Trade Matrix',
    'EI': 'Emissions',
    'GT': 'Temperature Change',
}


class FAOSTATFetcher(BaseFetcher):
    """
    Fetcher for FAOSTAT data.

    Example:
        >>> fetcher = FAOSTATFetcher()
        >>> # Fetch wheat production for USA
        >>> result = fetcher.fetch(
        ...     "QCL",
        ...     start_date="2000-01-01",
        ...     area_code=231,  # USA
        ...     item_code=15,   # Wheat
        ...     element_code=5510,  # Production
        ... )
    """

    BASE_URL = "https://fenixservices.fao.org/faostat/api/v1"

    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=30,
        min_delay_seconds=2.0,
    )

    def __init__(
        self,
        cache: Optional[Any] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        if requests is None:
            raise ImportError("requests library required. Install with: pip install requests")

        super().__init__(
            api_key=None,
            cache=cache,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        return "faostat"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch FAOSTAT data.

        Args:
            series_id: Domain code (e.g., 'QCL', 'FBS')
            start_date: Start year
            end_date: End year
            **kwargs:
                - area_code: Country code (e.g., 231 for USA)
                - item_code: Item code (e.g., 15 for wheat)
                - element_code: Element code (e.g., 5510 for production)
        """
        domain = series_id

        # Build query parameters
        params = {
            'area': kwargs.get('area_code', ''),
            'item': kwargs.get('item_code', ''),
            'element': kwargs.get('element_code', ''),
        }

        if start_date:
            params['year_start'] = start_date.year
        if end_date:
            params['year_end'] = end_date.year

        # Remove empty params
        params = {k: v for k, v in params.items() if v}

        url = f"{self.BASE_URL}/en/data/{domain}"

        try:
            response = self._session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"FAOSTAT request failed: {e}")

        # Parse response
        records = data.get('data', [])
        if not records:
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        rows = []
        for rec in records:
            rows.append({
                'date': pd.to_datetime(f"{rec.get('Year', 2000)}-01-01"),
                'value': rec.get('Value'),
                'series_id': f"{domain}_{rec.get('Item Code', '')}_{rec.get('Element Code', '')}",
                'area': rec.get('Area'),
                'item': rec.get('Item'),
                'element': rec.get('Element'),
                'unit': rec.get('Unit'),
            })

        return pd.DataFrame(rows)

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available FAOSTAT domains."""
        return [
            {'id': code, 'name': name, 'source': 'faostat'}
            for code, name in FAOSTAT_DOMAINS.items()
        ]


__all__ = ['FAOSTATFetcher', 'FAOSTAT_DOMAINS']
