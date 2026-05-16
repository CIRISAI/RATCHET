"""
V-Dem Fetcher - Varieties of Democracy

Loads V-Dem democracy indices from downloaded datasets.
V-Dem does not provide a REST API; data must be downloaded manually.

Download: https://www.v-dem.net/data/the-v-dem-dataset/

Key variables for RATCHET:
    - v2x_polyarchy: Electoral Democracy Index
    - v2x_libdem: Liberal Democracy Index
    - v2x_partipdem: Participatory Democracy Index
    - v2x_delibdem: Deliberative Democracy Index
    - v2x_egaldem: Egalitarian Democracy Index
    - v2x_corr: Political Corruption Index
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base import BaseFetcher, FetchError

logger = logging.getLogger(__name__)


# Key V-Dem variables
VDEM_VARIABLES = {
    'v2x_polyarchy': 'Electoral Democracy Index',
    'v2x_libdem': 'Liberal Democracy Index',
    'v2x_partipdem': 'Participatory Democracy Index',
    'v2x_delibdem': 'Deliberative Democracy Index',
    'v2x_egaldem': 'Egalitarian Democracy Index',
    'v2x_corr': 'Political Corruption Index',
    'v2x_rule': 'Rule of Law Index',
    'v2xcl_disc': 'Freedom of Discussion',
    'v2x_freexp_altinf': 'Freedom of Expression',
    'v2x_frassoc_thick': 'Freedom of Association',
    'v2x_suffr': 'Share of Population with Suffrage',
    'v2xel_frefair': 'Free and Fair Elections',
    'v2x_elecoff': 'Elected Officials Index',
}


class VDemFetcher(BaseFetcher):
    """
    Fetcher for V-Dem democracy data.

    Since V-Dem doesn't have an API, this fetcher loads from local files.
    Download the V-Dem dataset from https://www.v-dem.net/data/

    Example:
        >>> fetcher = VDemFetcher(data_path="path/to/vdem.csv")
        >>> result = fetcher.fetch(
        ...     "v2x_polyarchy",
        ...     start_date="1990-01-01",
        ...     country="United States",
        ... )
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        cache: Optional[Any] = None,
    ):
        """
        Initialize V-Dem fetcher.

        Args:
            data_path: Path to V-Dem CSV file
            cache: Cache instance
        """
        super().__init__(api_key=None, cache=cache)

        self.data_path = Path(data_path) if data_path else None
        self._data: Optional[pd.DataFrame] = None

    @property
    def source_name(self) -> str:
        return "vdem"

    def _load_data(self) -> pd.DataFrame:
        """Load V-Dem data from file."""
        if self._data is not None:
            return self._data

        if self.data_path is None or not self.data_path.exists():
            raise FetchError(
                f"V-Dem data file not found at {self.data_path}. "
                "Download from https://www.v-dem.net/data/"
            )

        logger.info(f"Loading V-Dem data from {self.data_path}")
        self._data = pd.read_csv(self.data_path, low_memory=False)
        return self._data

    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fetch V-Dem variable.

        Args:
            series_id: V-Dem variable name (e.g., 'v2x_polyarchy')
            start_date: Start year
            end_date: End year
            **kwargs:
                - country: Country name
                - country_code: Country code
        """
        data = self._load_data()

        if series_id not in data.columns:
            raise FetchError(f"Variable {series_id} not found in V-Dem data")

        # Filter data
        result = data[['country_name', 'country_text_id', 'year', series_id]].copy()

        # Apply filters
        if 'country' in kwargs:
            result = result[result['country_name'] == kwargs['country']]
        if 'country_code' in kwargs:
            result = result[result['country_text_id'] == kwargs['country_code']]

        if start_date:
            result = result[result['year'] >= start_date.year]
        if end_date:
            result = result[result['year'] <= end_date.year]

        # Format output
        output = pd.DataFrame({
            'date': pd.to_datetime(result['year'].astype(str) + '-01-01'),
            'value': result[series_id].values,
            'series_id': series_id,
            'country': result['country_name'].values,
            'country_code': result['country_text_id'].values,
        })

        return output

    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available V-Dem variables."""
        variables = [
            {'id': var, 'name': name, 'source': 'vdem'}
            for var, name in VDEM_VARIABLES.items()
        ]

        if search:
            search_lower = search.lower()
            variables = [
                v for v in variables
                if search_lower in v['id'].lower() or search_lower in v['name'].lower()
            ]

        return variables


__all__ = ['VDemFetcher', 'VDEM_VARIABLES']
