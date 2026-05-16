"""
Temporal Alignment for RATCHET Data Pipeline

Aligns data from different sources to a common time scale.
Handles resampling, interpolation, and aggregation across different frequencies.

Supported frequencies:
    - Daily (D)
    - Weekly (W)
    - Monthly (M)
    - Quarterly (Q)
    - Yearly (Y)
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Frequency(Enum):
    """Data frequency levels."""
    DAILY = 'D'
    WEEKLY = 'W'
    MONTHLY = 'M'
    QUARTERLY = 'Q'
    YEARLY = 'Y'

    @property
    def pandas_freq(self) -> str:
        """Get pandas frequency string."""
        mapping = {
            'D': 'D',
            'W': 'W',
            'M': 'ME',  # Month End
            'Q': 'QE',  # Quarter End
            'Y': 'YE',  # Year End
        }
        return mapping[self.value]

    @property
    def days_per_period(self) -> int:
        """Approximate days per period."""
        mapping = {
            'D': 1,
            'W': 7,
            'M': 30,
            'Q': 91,
            'Y': 365,
        }
        return mapping[self.value]


class AggregationMethod(Enum):
    """Methods for aggregating data when downsampling."""
    MEAN = 'mean'
    SUM = 'sum'
    LAST = 'last'
    FIRST = 'first'
    MIN = 'min'
    MAX = 'max'
    MEDIAN = 'median'


class TemporalAligner:
    """
    Aligns time series data to a common frequency.

    Handles:
        - Upsampling (daily -> monthly) via aggregation
        - Downsampling (monthly -> daily) via interpolation
        - Mixed frequency alignment across multiple series
        - Gap detection and filling

    Example:
        >>> aligner = TemporalAligner(target_frequency=Frequency.MONTHLY)
        >>> aligned = aligner.align(df, date_column='date', value_column='value')
        >>> merged = aligner.merge_series([df1, df2, df3])
    """

    def __init__(
        self,
        target_frequency: Frequency = Frequency.MONTHLY,
        aggregation_method: AggregationMethod = AggregationMethod.MEAN,
        interpolation_method: str = 'linear',
        fill_gaps: bool = True,
        max_gap_days: int = 90,
    ):
        """
        Initialize the aligner.

        Args:
            target_frequency: Target frequency for alignment
            aggregation_method: Method for aggregating when downsampling
            interpolation_method: Method for interpolation when upsampling
                Options: 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            fill_gaps: Whether to fill gaps in data
            max_gap_days: Maximum gap to fill (larger gaps left as NaN)
        """
        self.target_frequency = target_frequency
        self.aggregation_method = aggregation_method
        self.interpolation_method = interpolation_method
        self.fill_gaps = fill_gaps
        self.max_gap_days = max_gap_days

    def detect_frequency(self, df: pd.DataFrame, date_column: str = 'date') -> Frequency:
        """
        Detect the frequency of a time series.

        Args:
            df: DataFrame with date column
            date_column: Name of date column

        Returns:
            Detected Frequency
        """
        if len(df) < 2:
            return self.target_frequency

        dates = pd.to_datetime(df[date_column]).sort_values()
        gaps = dates.diff().dropna()
        median_gap_days = gaps.dt.days.median()

        if median_gap_days <= 2:
            return Frequency.DAILY
        elif median_gap_days <= 10:
            return Frequency.WEEKLY
        elif median_gap_days <= 45:
            return Frequency.MONTHLY
        elif median_gap_days <= 120:
            return Frequency.QUARTERLY
        else:
            return Frequency.YEARLY

    def align(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'value',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Align a time series to the target frequency.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column
            start_date: Start of output range (default: data min)
            end_date: End of output range (default: data max)

        Returns:
            DataFrame with aligned data
        """
        if df.empty:
            return df

        # Ensure date column is datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date
        df = df.sort_values(date_column)

        # Detect source frequency
        source_freq = self.detect_frequency(df, date_column)

        # Set date as index
        df_indexed = df.set_index(date_column)

        # Determine if we need to upsample or downsample
        source_days = source_freq.days_per_period
        target_days = self.target_frequency.days_per_period

        if source_days < target_days:
            # Downsample (aggregate)
            aligned = self._downsample(df_indexed, value_column)
        elif source_days > target_days:
            # Upsample (interpolate)
            aligned = self._upsample(df_indexed, value_column, start_date, end_date)
        else:
            # Same frequency, just resample to align dates
            aligned = df_indexed[[value_column]].resample(
                self.target_frequency.pandas_freq
            ).agg(self.aggregation_method.value)

        # Fill gaps if requested
        if self.fill_gaps:
            aligned = self._fill_gaps(aligned, value_column)

        # Reset index to get date as column
        aligned = aligned.reset_index()
        aligned.columns = [date_column, value_column]

        return aligned

    def _downsample(
        self,
        df: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        """Downsample data by aggregation."""
        return df[[value_column]].resample(
            self.target_frequency.pandas_freq
        ).agg(self.aggregation_method.value)

    def _upsample(
        self,
        df: pd.DataFrame,
        value_column: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Upsample data by interpolation."""
        # Create target date range
        if start_date is None:
            start_date = df.index.min()
        if end_date is None:
            end_date = df.index.max()

        target_index = pd.date_range(
            start=start_date,
            end=end_date,
            freq=self.target_frequency.pandas_freq,
        )

        # Reindex and interpolate
        upsampled = df[[value_column]].reindex(
            df.index.union(target_index)
        )

        upsampled = upsampled.interpolate(
            method=self.interpolation_method,
            limit_direction='both',
        )

        # Keep only target dates
        upsampled = upsampled.reindex(target_index)

        return upsampled

    def _fill_gaps(
        self,
        df: pd.DataFrame,
        value_column: str,
    ) -> pd.DataFrame:
        """Fill gaps in data up to max_gap_days."""
        if df.empty or value_column not in df.columns:
            return df

        # Find gap sizes
        values = df[value_column].values
        is_nan = pd.isna(values)

        # Calculate consecutive NaN counts
        gap_sizes = []
        current_gap = 0
        for i, nan in enumerate(is_nan):
            if nan:
                current_gap += 1
            else:
                if current_gap > 0:
                    gap_sizes.append((i - current_gap, i - 1, current_gap))
                current_gap = 0
        if current_gap > 0:
            gap_sizes.append((len(values) - current_gap, len(values) - 1, current_gap))

        # Estimate days per period for gap size check
        days_per_period = self.target_frequency.days_per_period
        max_gap_periods = self.max_gap_days // days_per_period

        # Fill small gaps
        for start_idx, end_idx, gap_size in gap_sizes:
            if gap_size <= max_gap_periods:
                # Interpolate this gap
                df[value_column] = df[value_column].interpolate(
                    method=self.interpolation_method,
                    limit=gap_size,
                    limit_direction='both',
                )

        return df

    def merge_series(
        self,
        series_list: List[pd.DataFrame],
        date_column: str = 'date',
        value_columns: Optional[List[str]] = None,
        how: str = 'outer',
    ) -> pd.DataFrame:
        """
        Merge multiple time series to a common time scale.

        Args:
            series_list: List of DataFrames to merge
            date_column: Name of date column (must be same in all)
            value_columns: Names for value columns (default: value_0, value_1, ...)
            how: Merge method ('inner', 'outer', 'left', 'right')

        Returns:
            Merged DataFrame with aligned dates
        """
        if not series_list:
            return pd.DataFrame()

        # Align each series
        aligned_list = []
        for i, df in enumerate(series_list):
            aligned = self.align(df, date_column=date_column)
            if value_columns and i < len(value_columns):
                aligned.columns = [date_column, value_columns[i]]
            else:
                aligned.columns = [date_column, f'value_{i}']
            aligned_list.append(aligned.set_index(date_column))

        # Merge all series
        if len(aligned_list) == 1:
            return aligned_list[0].reset_index()

        merged = aligned_list[0]
        for df in aligned_list[1:]:
            merged = merged.join(df, how=how)

        return merged.reset_index()

    def detect_gaps(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'value',
    ) -> List[Dict[str, Any]]:
        """
        Detect gaps in a time series.

        Args:
            df: DataFrame to analyze
            date_column: Name of date column
            value_column: Name of value column

        Returns:
            List of gap descriptions
        """
        if df.empty:
            return []

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        gaps = []

        # Check for missing dates
        dates = df[date_column]
        expected_freq = self.detect_frequency(df, date_column)

        expected_dates = pd.date_range(
            start=dates.min(),
            end=dates.max(),
            freq=expected_freq.pandas_freq,
        )

        missing_dates = expected_dates.difference(dates)
        if len(missing_dates) > 0:
            gaps.append({
                'type': 'missing_dates',
                'count': len(missing_dates),
                'dates': missing_dates.tolist(),
            })

        # Check for NaN values
        nan_count = df[value_column].isna().sum()
        if nan_count > 0:
            nan_dates = df.loc[df[value_column].isna(), date_column].tolist()
            gaps.append({
                'type': 'nan_values',
                'count': nan_count,
                'dates': nan_dates,
            })

        return gaps

    def get_coverage(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        value_column: str = 'value',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Calculate data coverage statistics.

        Args:
            df: DataFrame to analyze
            date_column: Name of date column
            value_column: Name of value column
            start_date: Start of analysis range
            end_date: End of analysis range

        Returns:
            Dict with coverage statistics
        """
        if df.empty:
            return {
                'total_periods': 0,
                'available_periods': 0,
                'coverage_pct': 0.0,
                'start_date': None,
                'end_date': None,
            }

        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Filter to range
        if start_date:
            df = df[df[date_column] >= start_date]
        if end_date:
            df = df[df[date_column] <= end_date]

        if df.empty:
            return {
                'total_periods': 0,
                'available_periods': 0,
                'coverage_pct': 0.0,
                'start_date': start_date,
                'end_date': end_date,
            }

        # Calculate expected periods
        actual_start = df[date_column].min()
        actual_end = df[date_column].max()

        expected_dates = pd.date_range(
            start=actual_start,
            end=actual_end,
            freq=self.target_frequency.pandas_freq,
        )

        total_periods = len(expected_dates)
        available = df[value_column].notna().sum()

        return {
            'total_periods': total_periods,
            'available_periods': int(available),
            'coverage_pct': (available / total_periods * 100) if total_periods > 0 else 0.0,
            'start_date': actual_start,
            'end_date': actual_end,
            'frequency': self.target_frequency.value,
        }


def align_to_monthly(
    df: pd.DataFrame,
    date_column: str = 'date',
    value_column: str = 'value',
) -> pd.DataFrame:
    """
    Convenience function to align data to monthly frequency.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        value_column: Name of value column

    Returns:
        Monthly-aligned DataFrame
    """
    aligner = TemporalAligner(target_frequency=Frequency.MONTHLY)
    return aligner.align(df, date_column, value_column)


def align_to_yearly(
    df: pd.DataFrame,
    date_column: str = 'date',
    value_column: str = 'value',
    aggregation: AggregationMethod = AggregationMethod.MEAN,
) -> pd.DataFrame:
    """
    Convenience function to align data to yearly frequency.

    Args:
        df: Input DataFrame
        date_column: Name of date column
        value_column: Name of value column
        aggregation: Aggregation method for yearly values

    Returns:
        Yearly-aligned DataFrame
    """
    aligner = TemporalAligner(
        target_frequency=Frequency.YEARLY,
        aggregation_method=aggregation,
    )
    return aligner.align(df, date_column, value_column)


__all__ = [
    'TemporalAligner',
    'Frequency',
    'AggregationMethod',
    'align_to_monthly',
    'align_to_yearly',
]
