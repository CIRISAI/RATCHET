"""
RATCHET Data Pipeline

Infrastructure for fetching, caching, and normalizing external data sources
for use with RATCHET engines.

Supported data sources:
    - FRED: Federal Reserve Economic Data
    - FAOSTAT: UN Food and Agriculture Organization
    - GDELT: Global Database of Events, Language, and Tone
    - V-Dem: Varieties of Democracy
    - IUCN: Red List of Threatened Species
    - UN COMTRADE: International Trade Statistics
    - OpenAlex: Academic publications and citations
"""

from .fetchers import BaseFetcher, FREDFetcher
from .cache import SQLiteCache
from .normalizers import TemporalAligner

__all__ = [
    'BaseFetcher',
    'FREDFetcher',
    'SQLiteCache',
    'TemporalAligner',
]
