"""
Abstract Base Fetcher for RATCHET Data Pipeline

Provides common functionality for all data fetchers:
    - Rate limiting with configurable delays
    - Exponential backoff retry logic
    - Cache integration interface
    - Standardized output format (DataFrame)
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class FetchError(Exception):
    """Base exception for fetch errors."""
    pass


class RateLimitError(FetchError):
    """Raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class DataQualityError(FetchError):
    """Raised when fetched data fails quality checks."""
    pass


class Frequency(Enum):
    """Data frequency levels for temporal alignment."""
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    YEARLY = 'Y'


@dataclass
class FetchResult:
    """Result of a fetch operation with metadata."""
    data: pd.DataFrame
    source: str
    series_id: str
    fetch_time: datetime = field(default_factory=datetime.utcnow)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    frequency: Optional[Frequency] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False

    @property
    def row_count(self) -> int:
        return len(self.data)

    @property
    def date_range(self) -> tuple:
        if self.data.empty:
            return (None, None)
        if 'date' in self.data.columns:
            return (self.data['date'].min(), self.data['date'].max())
        return (self.start_date, self.end_date)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source,
            'series_id': self.series_id,
            'row_count': self.row_count,
            'date_range': self.date_range,
            'frequency': self.frequency.value if self.frequency else None,
            'from_cache': self.from_cache,
            'fetch_time': self.fetch_time.isoformat(),
        }


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_day: Optional[int] = None
    min_delay_seconds: float = 0.1
    burst_limit: int = 10


@dataclass
class RetryConfig:
    """Retry configuration with exponential backoff."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_errors: tuple = (ConnectionError, TimeoutError, RateLimitError)


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers.

    Provides rate limiting, retry logic, and cache integration.
    Subclasses must implement:
        - _fetch_impl: actual data fetching logic
        - source_name: property returning source identifier

    Example:
        >>> class MyFetcher(BaseFetcher):
        ...     @property
        ...     def source_name(self) -> str:
        ...         return "my_source"
        ...
        ...     def _fetch_impl(self, series_id, start_date, end_date):
        ...         # Fetch data from API
        ...         return pd.DataFrame(...)
        ...
        >>> fetcher = MyFetcher(api_key="...")
        >>> result = fetcher.fetch("GDP", start_date="2020-01-01")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[Any] = None,  # SQLiteCache
        rate_limit: Optional[RateLimitConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize the fetcher.

        Args:
            api_key: API key for the data source
            cache: Cache instance for storing fetched data
            rate_limit: Rate limiting configuration
            retry_config: Retry configuration
        """
        self.api_key = api_key
        self.cache = cache
        self.rate_limit = rate_limit or RateLimitConfig()
        self.retry_config = retry_config or RetryConfig()

        # Rate limiting state
        self._request_times: List[float] = []
        self._daily_request_count: int = 0
        self._daily_reset_time: datetime = datetime.utcnow()

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the source identifier (e.g., 'fred', 'faostat')."""
        pass

    @abstractmethod
    def _fetch_impl(
        self,
        series_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Implement the actual data fetching logic.

        Args:
            series_id: Identifier for the data series
            start_date: Start date for data range
            end_date: End date for data range
            **kwargs: Additional source-specific parameters

        Returns:
            DataFrame with at least 'date' and 'value' columns
        """
        pass

    @abstractmethod
    def list_series(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available data series.

        Args:
            search: Optional search term to filter series

        Returns:
            List of dicts with series metadata
        """
        pass

    def fetch(
        self,
        series_id: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> FetchResult:
        """
        Fetch data for a series with caching, rate limiting, and retries.

        Args:
            series_id: Identifier for the data series
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            use_cache: Whether to use cached data if available
            **kwargs: Additional source-specific parameters

        Returns:
            FetchResult with data and metadata

        Raises:
            FetchError: If fetch fails after all retries
            RateLimitError: If rate limit exceeded and cannot wait
        """
        # Parse dates
        start_dt = self._parse_date(start_date)
        end_dt = self._parse_date(end_date) or datetime.utcnow()

        # Check cache first
        if use_cache and self.cache is not None:
            cached = self.cache.get(
                source=self.source_name,
                series_id=series_id,
                start_date=start_dt,
                end_date=end_dt,
            )
            if cached is not None:
                logger.debug(f"Cache hit for {self.source_name}/{series_id}")
                return FetchResult(
                    data=cached,
                    source=self.source_name,
                    series_id=series_id,
                    start_date=start_dt,
                    end_date=end_dt,
                    from_cache=True,
                )

        # Apply rate limiting
        self._apply_rate_limit()

        # Fetch with retry logic
        last_error = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                data = self._fetch_impl(series_id, start_dt, end_dt, **kwargs)

                # Store in cache
                if self.cache is not None:
                    self.cache.put(
                        source=self.source_name,
                        series_id=series_id,
                        data=data,
                        start_date=start_dt,
                        end_date=end_dt,
                    )

                # Detect frequency
                frequency = self._detect_frequency(data)

                return FetchResult(
                    data=data,
                    source=self.source_name,
                    series_id=series_id,
                    start_date=start_dt,
                    end_date=end_dt,
                    frequency=frequency,
                    from_cache=False,
                )

            except self.retry_config.retryable_errors as e:
                last_error = e
                if attempt < self.retry_config.max_retries:
                    delay = self._compute_backoff_delay(attempt)
                    logger.warning(
                        f"Fetch attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

        raise FetchError(
            f"Failed to fetch {series_id} after {self.retry_config.max_retries + 1} "
            f"attempts. Last error: {last_error}"
        )

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting, waiting if necessary."""
        now = time.time()

        # Reset daily counter if needed
        if datetime.utcnow() - self._daily_reset_time > timedelta(days=1):
            self._daily_request_count = 0
            self._daily_reset_time = datetime.utcnow()

        # Check daily limit
        if self.rate_limit.requests_per_day is not None:
            if self._daily_request_count >= self.rate_limit.requests_per_day:
                raise RateLimitError(
                    f"Daily limit of {self.rate_limit.requests_per_day} requests exceeded",
                    retry_after=self._seconds_until_daily_reset(),
                )

        # Clean old request times (keep last minute)
        minute_ago = now - 60
        self._request_times = [t for t in self._request_times if t > minute_ago]

        # Check per-minute limit
        if len(self._request_times) >= self.rate_limit.requests_per_minute:
            # Wait until oldest request is older than 1 minute
            wait_time = self._request_times[0] + 60 - now
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        # Apply minimum delay
        if self._request_times:
            time_since_last = now - self._request_times[-1]
            if time_since_last < self.rate_limit.min_delay_seconds:
                time.sleep(self.rate_limit.min_delay_seconds - time_since_last)

        # Record this request
        self._request_times.append(time.time())
        self._daily_request_count += 1

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute delay for exponential backoff."""
        import random

        delay = self.retry_config.initial_delay * (
            self.retry_config.exponential_base ** attempt
        )
        delay = min(delay, self.retry_config.max_delay)

        if self.retry_config.jitter:
            delay *= (0.5 + random.random())

        return delay

    def _seconds_until_daily_reset(self) -> float:
        """Calculate seconds until daily counter resets."""
        reset_time = self._daily_reset_time + timedelta(days=1)
        return (reset_time - datetime.utcnow()).total_seconds()

    def _parse_date(self, date: Optional[Union[str, datetime]]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if date is None:
            return None
        if isinstance(date, datetime):
            return date

        # Try common date formats
        formats = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d', '%d-%m-%Y', '%m/%d/%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date, fmt)
            except ValueError:
                continue

        raise ValueError(f"Could not parse date: {date}")

    def _detect_frequency(self, data: pd.DataFrame) -> Optional[Frequency]:
        """Detect data frequency from date column."""
        if data.empty or 'date' not in data.columns:
            return None

        dates = pd.to_datetime(data['date']).sort_values()
        if len(dates) < 2:
            return None

        # Calculate median gap between observations
        gaps = dates.diff().dropna()
        median_gap = gaps.median()

        if median_gap <= timedelta(days=2):
            return Frequency.DAILY
        elif median_gap <= timedelta(days=10):
            return Frequency.WEEKLY
        elif median_gap <= timedelta(days=45):
            return Frequency.MONTHLY
        elif median_gap <= timedelta(days=120):
            return Frequency.QUARTERLY
        else:
            return Frequency.YEARLY

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate fetched data quality.

        Override in subclasses for source-specific validation.

        Args:
            data: DataFrame to validate

        Returns:
            True if data passes validation

        Raises:
            DataQualityError: If validation fails
        """
        if data.empty:
            raise DataQualityError("Fetched data is empty")

        if 'date' not in data.columns:
            raise DataQualityError("Data missing required 'date' column")

        if 'value' not in data.columns:
            raise DataQualityError("Data missing required 'value' column")

        # Check for excessive missing values
        missing_pct = data['value'].isna().mean()
        if missing_pct > 0.5:
            raise DataQualityError(
                f"Data has {missing_pct:.1%} missing values (threshold: 50%)"
            )

        return True


__all__ = [
    'BaseFetcher',
    'FetchResult',
    'FetchError',
    'RateLimitError',
    'DataQualityError',
    'Frequency',
    'RateLimitConfig',
    'RetryConfig',
]
