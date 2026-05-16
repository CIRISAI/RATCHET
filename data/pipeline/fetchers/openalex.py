"""
OpenAlex Fetcher

Fetches academic publication and citation data from OpenAlex.
API Documentation: https://docs.openalex.org/

No API key required, but polite pool recommended (include email in user-agent).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

from .base import BaseFetcher, FetchError, RateLimitConfig

logger = logging.getLogger(__name__)


class OpenAlexFetcher(BaseFetcher):
    """
    Fetcher for OpenAlex academic data.

    Example:
        >>> fetcher = OpenAlexFetcher(email="your@email.com")
        >>> result = fetcher.fetch(
        ...     "works",
        ...     start_date="2020-01-01",
        ...     search="climate change",
        ... )
    """

    BASE_URL = "https://api.openalex.org"

    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=100,  # Polite pool allows more
        min_delay_seconds=0.1,
    )

    def __init__(
        self,
        email: Optional[str] = None,
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
        user_agent = 'RATCHET-DataPipeline/1.0'
        if email:
            user_agent += f' (mailto:{email})'
        self._session.headers.update({'User-Agent': user_agent})
        self.email = email

    @property
    def source_name(self) -> str:
        return "openalex"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch OpenAlex data.

        Args:
            series_id: Entity type ('works', 'authors', 'institutions', 'concepts')
            start_date: Start date
            end_date: End date
            **kwargs:
                - search: Search query
                - filter: Filter expression
                - group_by: Grouping field for aggregation
        """
        endpoint = f"{self.BASE_URL}/{series_id}"

        # Build filters
        filters = []
        if start_date:
            filters.append(f"from_publication_date:{start_date.strftime('%Y-%m-%d')}")
        if end_date:
            filters.append(f"to_publication_date:{end_date.strftime('%Y-%m-%d')}")
        if 'filter' in kwargs:
            filters.append(kwargs['filter'])

        params = {}
        if filters:
            params['filter'] = ','.join(filters)
        if 'search' in kwargs:
            params['search'] = kwargs['search']
        if 'group_by' in kwargs:
            params['group_by'] = kwargs['group_by']

        params['per-page'] = kwargs.get('per_page', 200)

        if self.email:
            params['mailto'] = self.email

        url = f"{endpoint}?{urlencode(params)}"

        try:
            response = self._session.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"OpenAlex request failed: {e}")

        # Handle grouped results
        if 'group_by' in data:
            return self._parse_grouped_results(data, series_id)

        # Handle list results
        results = data.get('results', [])
        if not results:
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        rows = []
        for item in results:
            pub_date = item.get('publication_date', '')
            rows.append({
                'date': pd.to_datetime(pub_date) if pub_date else None,
                'value': item.get('cited_by_count', 0),
                'series_id': f"openalex_{series_id}",
                'id': item.get('id'),
                'title': item.get('title', '')[:200] if item.get('title') else '',
                'doi': item.get('doi'),
                'type': item.get('type'),
                'cited_by_count': item.get('cited_by_count', 0),
            })

        return pd.DataFrame(rows)

    def _parse_grouped_results(
        self,
        data: Dict,
        series_id: str,
    ) -> pd.DataFrame:
        """Parse grouped/aggregated results."""
        groups = data.get('group_by', [])
        rows = []

        for group in groups:
            rows.append({
                'date': datetime.utcnow(),
                'value': group.get('count', 0),
                'series_id': f"openalex_{series_id}_grouped",
                'key': group.get('key'),
                'key_display_name': group.get('key_display_name'),
                'count': group.get('count', 0),
            })

        return pd.DataFrame(rows)

    def fetch_publication_counts(
        self,
        search: Optional[str] = None,
        start_year: int = 2000,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch publication counts by year.

        Args:
            search: Optional search term
            start_year: Start year
            end_year: End year (default: current year)

        Returns:
            DataFrame with yearly publication counts
        """
        if end_year is None:
            end_year = datetime.utcnow().year

        params = {
            'group_by': 'publication_year',
            'filter': f'publication_year:{start_year}-{end_year}',
        }
        if search:
            params['search'] = search

        result = self.fetch('works', **params)

        # Convert to time series format
        if result.empty:
            return result

        output = pd.DataFrame({
            'date': pd.to_datetime(result['key'].astype(str) + '-01-01'),
            'value': result['count'],
            'series_id': f"openalex_publications_{'_'.join(search.split()[:3]) if search else 'all'}",
        })

        return output.sort_values('date')

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available entity types."""
        return [
            {'id': 'works', 'name': 'Academic Works/Publications'},
            {'id': 'authors', 'name': 'Authors'},
            {'id': 'institutions', 'name': 'Research Institutions'},
            {'id': 'concepts', 'name': 'Research Concepts/Topics'},
            {'id': 'sources', 'name': 'Journals and Repositories'},
            {'id': 'publishers', 'name': 'Publishers'},
        ]


__all__ = ['OpenAlexFetcher']
