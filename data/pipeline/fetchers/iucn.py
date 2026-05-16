"""
IUCN Red List Fetcher

Fetches species conservation data from IUCN Red List API.
API Documentation: https://apiv3.iucnredlist.org/api/v3/docs

Requires API key from: https://apiv3.iucnredlist.org/api/v3/token
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


# IUCN Red List categories
IUCN_CATEGORIES = {
    'EX': 'Extinct',
    'EW': 'Extinct in the Wild',
    'CR': 'Critically Endangered',
    'EN': 'Endangered',
    'VU': 'Vulnerable',
    'NT': 'Near Threatened',
    'LC': 'Least Concern',
    'DD': 'Data Deficient',
    'NE': 'Not Evaluated',
}


class IUCNFetcher(BaseFetcher):
    """
    Fetcher for IUCN Red List data.

    Example:
        >>> fetcher = IUCNFetcher(api_key="your_key")
        >>> result = fetcher.fetch(
        ...     "species",
        ...     taxon_name="Panthera tigris",  # Tiger
        ... )
    """

    BASE_URL = "https://apiv3.iucnredlist.org/api/v3"

    DEFAULT_RATE_LIMIT = RateLimitConfig(
        requests_per_minute=30,
        min_delay_seconds=2.0,
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ):
        if requests is None:
            raise ImportError("requests library required")

        api_key = api_key or os.environ.get('IUCN_API_KEY')
        if not api_key:
            raise ValueError(
                "IUCN API key required. Set IUCN_API_KEY environment variable "
                "or get key from https://apiv3.iucnredlist.org/api/v3/token"
            )

        super().__init__(
            api_key=api_key,
            cache=cache,
            rate_limit=rate_limit or self.DEFAULT_RATE_LIMIT,
        )
        self._session = requests.Session()

    @property
    def source_name(self) -> str:
        return "iucn"

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch IUCN data.

        Args:
            series_id: Query type ('species', 'country', 'region')
            **kwargs:
                - taxon_name: Species scientific name
                - country_code: ISO country code
                - region: Region identifier
        """
        if series_id == 'species':
            return self._fetch_species(kwargs.get('taxon_name', ''))
        elif series_id == 'country':
            return self._fetch_country(kwargs.get('country_code', ''))
        else:
            raise FetchError(f"Unknown query type: {series_id}")

    def _fetch_species(self, taxon_name: str) -> pd.DataFrame:
        """Fetch species data by scientific name."""
        url = f"{self.BASE_URL}/species/{taxon_name}"
        params = {'token': self.api_key}

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"IUCN request failed: {e}")

        result = data.get('result', [])
        if not result:
            return pd.DataFrame(columns=['date', 'value', 'series_id'])

        rows = []
        for species in result:
            rows.append({
                'date': pd.to_datetime(species.get('assessment_date', '2000-01-01')),
                'value': species.get('category', ''),
                'series_id': f"iucn_species_{taxon_name}",
                'taxon_id': species.get('taxonid'),
                'scientific_name': species.get('scientific_name'),
                'category': species.get('category'),
                'population_trend': species.get('population_trend'),
            })

        return pd.DataFrame(rows)

    def _fetch_country(self, country_code: str) -> pd.DataFrame:
        """Fetch species count by country."""
        url = f"{self.BASE_URL}/country/getspecies/{country_code}"
        params = {'token': self.api_key}

        try:
            response = self._session.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            raise FetchError(f"IUCN request failed: {e}")

        result = data.get('result', [])

        # Aggregate by category
        category_counts = {}
        for species in result:
            cat = species.get('category', 'NE')
            category_counts[cat] = category_counts.get(cat, 0) + 1

        rows = []
        for cat, count in category_counts.items():
            rows.append({
                'date': datetime.utcnow(),
                'value': count,
                'series_id': f"iucn_country_{country_code}_{cat}",
                'country_code': country_code,
                'category': cat,
                'category_name': IUCN_CATEGORIES.get(cat, cat),
            })

        return pd.DataFrame(rows)

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available query types."""
        return [
            {'id': 'species', 'name': 'Species by scientific name'},
            {'id': 'country', 'name': 'Species by country'},
            {'id': 'region', 'name': 'Species by region'},
        ]


__all__ = ['IUCNFetcher', 'IUCN_CATEGORIES']
