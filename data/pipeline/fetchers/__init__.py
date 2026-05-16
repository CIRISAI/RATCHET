"""
Data Fetchers for RATCHET Pipeline

Each fetcher implements a common interface for retrieving data from external sources
with rate limiting, retry logic, and caching support.

Supported Sources:
    - FRED: Federal Reserve Economic Data
    - FAOSTAT: UN Food and Agriculture Organization
    - GDELT: Global Database of Events, Language, and Tone
    - V-Dem: Varieties of Democracy
    - IUCN: Red List of Threatened Species
    - COMTRADE: UN International Trade Statistics
    - OpenAlex: Academic publications and citations
"""

from .base import BaseFetcher, FetchResult, RateLimitError, FetchError, Frequency
from .fred import FREDFetcher
from .faostat import FAOSTATFetcher
from .gdelt import GDELTFetcher
from .vdem import VDemFetcher
from .iucn import IUCNFetcher
from .comtrade import COMTRADEFetcher
from .openalex import OpenAlexFetcher

__all__ = [
    # Base classes
    'BaseFetcher',
    'FetchResult',
    'RateLimitError',
    'FetchError',
    'Frequency',
    # Fetchers
    'FREDFetcher',
    'FAOSTATFetcher',
    'GDELTFetcher',
    'VDemFetcher',
    'IUCNFetcher',
    'COMTRADEFetcher',
    'OpenAlexFetcher',
]
