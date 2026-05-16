"""
Federal Reserve Economic Data (FRED) Fetcher

Fetches economic time series data from the FRED API.
Supports GDP, unemployment, inflation, interest rates, and thousands of other series.

API Documentation: https://fred.stlouisfed.org/docs/api/fred/

Relevant series for RATCHET:
    - GDP: Gross Domestic Product
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - FEDFUNDS: Federal Funds Rate
    - M2SL: M2 Money Stock
    - GFDEBTN: Federal Debt Total Public Debt
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

from .base import (
    BaseFetcher,
    FetchError,
    RateLimitError,
    RateLimitConfig,
    RetryConfig,
    Frequency,
)

logger = logging.getLogger(__name__)


# Common FRED series for RATCHET analysis
FRED_SERIES_CATALOG = {
    # GDP and Output
    'GDP': {'name': 'Gross Domestic Product', 'frequency': 'Q', 'units': 'Billions of Dollars'},
    'GDPC1': {'name': 'Real GDP', 'frequency': 'Q', 'units': 'Billions of Chained 2017 Dollars'},
    'A191RL1Q225SBEA': {'name': 'Real GDP Growth Rate', 'frequency': 'Q', 'units': 'Percent Change'},

    # Labor Market
    'UNRATE': {'name': 'Unemployment Rate', 'frequency': 'M', 'units': 'Percent'},
    'PAYEMS': {'name': 'Total Nonfarm Payrolls', 'frequency': 'M', 'units': 'Thousands'},
    'CIVPART': {'name': 'Labor Force Participation Rate', 'frequency': 'M', 'units': 'Percent'},

    # Prices and Inflation
    'CPIAUCSL': {'name': 'Consumer Price Index', 'frequency': 'M', 'units': 'Index 1982-84=100'},
    'CPILFESL': {'name': 'Core CPI', 'frequency': 'M', 'units': 'Index 1982-84=100'},
    'PCEPI': {'name': 'PCE Price Index', 'frequency': 'M', 'units': 'Index 2017=100'},

    # Interest Rates
    'FEDFUNDS': {'name': 'Federal Funds Rate', 'frequency': 'M', 'units': 'Percent'},
    'DGS10': {'name': '10-Year Treasury Rate', 'frequency': 'D', 'units': 'Percent'},
    'T10Y2Y': {'name': '10Y-2Y Yield Spread', 'frequency': 'D', 'units': 'Percent'},

    # Money and Credit
    'M2SL': {'name': 'M2 Money Stock', 'frequency': 'M', 'units': 'Billions of Dollars'},
    'TOTBKCR': {'name': 'Total Bank Credit', 'frequency': 'W', 'units': 'Billions of Dollars'},

    # Government Finance
    'GFDEBTN': {'name': 'Federal Debt Total', 'frequency': 'Q', 'units': 'Millions of Dollars'},
    'FYFSD': {'name': 'Federal Surplus/Deficit', 'frequency': 'Y', 'units': 'Millions of Dollars'},

    # Trade
    'BOPGSTB': {'name': 'Trade Balance', 'frequency': 'M', 'units': 'Millions of Dollars'},

    # Financial Conditions
    'VIXCLS': {'name': 'VIX Volatility Index', 'frequency': 'D', 'units': 'Index'},
    'BAMLH0A0HYM2': {'name': 'High Yield OAS', 'frequency': 'D', 'units': 'Percent'},
}


class FREDFetcher(BaseFetcher):
    """
    Fetcher for Federal Reserve Economic Data (FRED).

    Requires an API key from https://fred.stlouisfed.org/docs/api/api_key.html

    Example:
        >>> fetcher = FREDFetcher(api_key="your_api_key")
        >>> result = fetcher.fetch("GDP", start_date="2000-01-01")
        >>> print(result.data.head())
        >>> print(f"Date range: {result.date_range}")

    Environment variable:
        Set FRED_API_KEY to avoid passing api_key explicitly.
    """

    BASE_URL = "https://api.stlouisfed.org/fred"

    # FRED has a limit of 120 requests per minute
    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=100,  # Leave headroom
        requests_per_day=None,  # No daily limit
        min_delay_seconds=0.5,
        burst_limit=10,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize FRED fetcher.

        Args:
            api_key: FRED API key. Falls back to FRED_API_KEY env var.
            cache: Cache instance for storing fetched data
            rate_limit: Rate limiting configuration
            retry_config: Retry configuration
        """
        if requests is None:
            raise ImportError(
                "requests library required for FRED fetcher. "
                "Install with: pip install requests"
            )

        # Get API key from environment if not provided
        api_key = api_key or os.environ.get('FRED_API_KEY')
        if not api_key:
            raise ValueError(
                "FRED API key required. Pass api_key parameter or set FRED_API_KEY "
                "environment variable. Get a key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        super().__init__(
            api_key=api_key,
            cache=cache,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
            retry_config=retry_config,
        )

        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'RATCHET-DataPipeline/1.0'})

    @property
    def source_name(self) -> str:
        return "fred"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch a FRED series.

        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters:
                - units: 'lin', 'chg', 'ch1', 'pch', 'pc1', 'pca', 'cch', 'cca', 'log'
                - frequency: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a'
                - aggregation_method: 'avg', 'sum', 'eop'

        Returns:
            DataFrame with 'date', 'value', and metadata columns
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
        }

        if start_date:
            params['observation_start'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['observation_end'] = end_date.strftime('%Y-%m-%d')

        # Optional parameters
        if 'units' in kwargs:
            params['units'] = kwargs['units']
        if 'frequency' in kwargs:
            params['frequency'] = kwargs['frequency']
        if 'aggregation_method' in kwargs:
            params['aggregation_method'] = kwargs['aggregation_method']

        url = f"{self.BASE_URL}/series/observations?{urlencode(params)}"

        try:
            response = self._session.get(url, timeout=30)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                raise RateLimitError(
                    f"FRED rate limit exceeded",
                    retry_after=float(retry_after),
                )

            response.raise_for_status()
            data = response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                error_msg = e.response.json().get('error_message', str(e))
                raise FetchError(f"FRED API error: {error_msg}")
            raise

        # Parse observations
        observations = data.get('observations', [])
        if not observations:
            logger.warning(f"No observations returned for series {series_id}")
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        records = []
        for obs in observations:
            value = obs.get('value', '.')
            if value == '.':  # FRED uses '.' for missing values
                value = None
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = None

            records.append({
                'date': pd.to_datetime(obs['date']),
                'value': value,
                'series_id': series_id,
            })

        df = pd.DataFrame(records)

        # Add metadata
        df['source'] = 'fred'
        df['realtime_start'] = data.get('realtime_start')
        df['realtime_end'] = data.get('realtime_end')

        return df

    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get metadata for a FRED series.

        Args:
            series_id: FRED series ID

        Returns:
            Dict with series metadata
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
        }

        url = f"{self.BASE_URL}/series?{urlencode(params)}"

        try:
            self._apply_rate_limit()
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            series_list = data.get('seriess', [])
            if not series_list:
                raise FetchError(f"Series {series_id} not found")

            return series_list[0]

        except requests.exceptions.HTTPError as e:
            raise FetchError(f"Failed to get series info: {e}")

    def list_series(
        self,
        search: Optional[str] = None,
        category_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List available FRED series.

        Args:
            search: Search term to filter series
            category_id: FRED category ID to browse
            limit: Maximum number of results

        Returns:
            List of dicts with series metadata
        """
        if search:
            # Use search endpoint
            params = {
                'search_text': search,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
            }
            url = f"{self.BASE_URL}/series/search?{urlencode(params)}"
        elif category_id:
            # Use category endpoint
            params = {
                'category_id': category_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'limit': limit,
            }
            url = f"{self.BASE_URL}/category/series?{urlencode(params)}"
        else:
            # Return common series from catalog
            return [
                {'id': k, **v}
                for k, v in FRED_SERIES_CATALOG.items()
            ]

        try:
            self._apply_rate_limit()
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get('seriess', [])

        except requests.exceptions.HTTPError as e:
            raise FetchError(f"Failed to list series: {e}")

    def fetch_multiple(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch multiple series and combine into a single DataFrame.

        Args:
            series_ids: List of FRED series IDs
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cache

        Returns:
            DataFrame with date index and series as columns
        """
        results = {}

        for series_id in series_ids:
            try:
                result = self.fetch(
                    series_id,
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=use_cache,
                )
                # Pivot to series name as column
                series_data = result.data[['date', 'value']].copy()
                series_data = series_data.set_index('date')
                series_data.columns = [series_id]
                results[series_id] = series_data

            except FetchError as e:
                logger.warning(f"Failed to fetch {series_id}: {e}")
                continue

        if not results:
            return pd.DataFrame()

        # Combine all series
        combined = pd.concat(results.values(), axis=1)
        combined = combined.sort_index()

        return combined

    def get_release_dates(self, series_id: str) -> List[datetime]:
        """
        Get release dates for a series (useful for real-time analysis).

        Args:
            series_id: FRED series ID

        Returns:
            List of release dates
        """
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
        }

        url = f"{self.BASE_URL}/series/release?{urlencode(params)}"

        try:
            self._apply_rate_limit()
            response = self._session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()

            releases = data.get('releases', [])
            dates = []
            for rel in releases:
                if 'realtime_start' in rel:
                    dates.append(pd.to_datetime(rel['realtime_start']))
            return dates

        except requests.exceptions.HTTPError as e:
            raise FetchError(f"Failed to get release dates: {e}")


def fetch_fred_data(
    series_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to fetch FRED data.

    Args:
        series_id: FRED series ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: FRED API key (or set FRED_API_KEY env var)

    Returns:
        DataFrame with date and value columns

    Example:
        >>> df = fetch_fred_data("GDP", "2000-01-01", "2023-12-31")
        >>> print(df.head())
    """
    fetcher = FREDFetcher(api_key=api_key)
    result = fetcher.fetch(series_id, start_date=start_date, end_date=end_date)
    return result.data


__all__ = [
    'FREDFetcher',
    'fetch_fred_data',
    'FRED_SERIES_CATALOG',
]
