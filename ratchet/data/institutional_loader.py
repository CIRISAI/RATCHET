"""
RATCHET Institutional Data Loader

Loads and preprocesses empirical institutional datasets for use with
InstitutionalCollapseEngine. Maps real-world variables to RATCHET framework.

Supported Datasets:
    - QoG Standard Time-Series (primary comprehensive dataset)
    - Polity V (regime transitions and executive constraints)
    - V-Dem (democracy indices, corruption)

Variable Mapping to RATCHET Framework:
    k (constraints):     Executive constraints (xconst from Polity, normalized)
    rho (coupling):      Elite concentration (derived from corruption indicators)
    sigma (stability):   Political stability (vdem_polyarchy, normalized)
    f (corruption):      Corruption level (vdem_corr, vdem_pubcorr)
    lambda (rule of law): Rule of law strength (wbgi_rle, normalized)

Collapse Event Detection:
    - Polity regtrans codes: -2 (adverse regime transition), -1 (negative change)
    - Large negative changes in polity2 score (> 6 points)
    - State failure indicators (sf column in Polity)
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd


class CollapseType(Enum):
    """Types of institutional collapse events."""
    ADVERSE_REGIME_TRANSITION = "adverse_regime_transition"  # Polity regtrans -2
    STATE_FAILURE = "state_failure"                          # Polity sf flag
    DEMOCRATIC_BREAKDOWN = "democratic_breakdown"            # Large negative polity change
    AUTOCRATIZATION = "autocratization"                      # Gradual decline
    COUP = "coup"                                            # Sudden regime change


@dataclass
class CollapseEvent:
    """Represents an institutional collapse or major transition event."""
    country: str
    country_code: str
    year: int
    collapse_type: CollapseType
    polity_change: Optional[float] = None
    polity_before: Optional[float] = None
    polity_after: Optional[float] = None
    regtrans_code: Optional[float] = None
    description: str = ""

    def __repr__(self) -> str:
        return (f"CollapseEvent({self.country}, {self.year}, "
                f"{self.collapse_type.value}, change={self.polity_change})")


@dataclass
class CountryTrajectory:
    """Time-series trajectory for a single country with RATCHET variables."""
    country: str
    country_code: str
    years: np.ndarray
    k: np.ndarray           # Constraints (0-1)
    rho: np.ndarray         # Elite coupling (0-1)
    sigma: np.ndarray       # Political stability (0-1)
    f: np.ndarray           # Corruption (0-1)
    lambda_: np.ndarray     # Rule of law (0-1)
    collapse_events: List[CollapseEvent] = field(default_factory=list)
    raw_data: Optional[pd.DataFrame] = None

    @property
    def duration(self) -> int:
        """Number of years in trajectory."""
        return len(self.years)

    @property
    def start_year(self) -> int:
        return int(self.years[0])

    @property
    def end_year(self) -> int:
        return int(self.years[-1])

    def get_state_at_year(self, year: int) -> Optional[Dict[str, float]]:
        """Get RATCHET state variables for a specific year."""
        idx = np.where(self.years == year)[0]
        if len(idx) == 0:
            return None
        i = idx[0]
        return {
            'k': float(self.k[i]),
            'rho': float(self.rho[i]),
            'sigma': float(self.sigma[i]),
            'f': float(self.f[i]),
            'lambda_': float(self.lambda_[i]),
            'year': year,
        }

    def has_collapse(self) -> bool:
        """Check if trajectory contains any collapse events."""
        return len(self.collapse_events) > 0

    def first_collapse_year(self) -> Optional[int]:
        """Get year of first collapse event, if any."""
        if not self.collapse_events:
            return None
        return min(e.year for e in self.collapse_events)

    def to_dataframe(self) -> pd.DataFrame:
        """Export trajectory as DataFrame."""
        return pd.DataFrame({
            'country': self.country,
            'country_code': self.country_code,
            'year': self.years,
            'k': self.k,
            'rho': self.rho,
            'sigma': self.sigma,
            'f': self.f,
            'lambda': self.lambda_,
        })


class InstitutionalDataLoader:
    """
    Loads and preprocesses institutional collapse datasets.

    Maps empirical data to RATCHET framework variables:
        k:      Constraints (executive constraints, checks and balances)
        rho:    Elite coupling (power concentration, neopatrimonialism proxy)
        sigma:  Political stability (democracy level, governance quality)
        f:      Corruption (public sector corruption)
        lambda: Rule of law (judicial independence, legal enforcement)

    Example:
        >>> loader = InstitutionalDataLoader('/path/to/data/institutional')
        >>> loader.load_all()
        >>> trajectory = loader.get_country_trajectory('Venezuela')
        >>> print(trajectory.collapse_events)
    """

    # Default data directory
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'institutional'

    # Expected dataset filenames
    QOG_FILENAME = 'qog_std_ts_jan25.csv'
    POLITY_FILENAME = 'p5v2018.csv'

    # Variable mappings
    QOG_VARS = {
        # V-Dem democracy/polyarchy
        'vdem_polyarchy': 'sigma_vdem',
        'vdem_libdem': 'sigma_libdem',
        # V-Dem corruption
        'vdem_corr': 'f_vdem',
        'vdem_pubcorr': 'f_pubcorr',
        'vdem_execorr': 'f_exec',
        # World Bank Governance Indicators
        'wbgi_rle': 'lambda_wbgi',  # Rule of Law
        'wbgi_cce': 'f_wbgi',       # Control of Corruption
        'wbgi_pve': 'sigma_wbgi',   # Political Stability
        # Polity
        'p_polity2': 'polity2',
        # ICRG Quality of Government
        'icrg_qog': 'qog_icrg',
    }

    POLITY_VARS = {
        'xconst': 'k_polity',       # Executive constraints (1-7)
        'polity2': 'polity2_raw',   # Polity score (-10 to 10)
        'regtrans': 'regtrans',     # Regime transition code
        'change': 'polity_change',  # Year-to-year change
        'sf': 'state_failure',      # State failure flag
        'durable': 'regime_durable', # Regime durability
    }

    # Country name standardization mapping
    COUNTRY_NAME_MAP = {
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Iran (Islamic Republic of)': 'Iran',
        'Korea (the Republic of)': 'South Korea',
        "Korea (the Democratic People's Republic of)": 'North Korea',
        'Russian Federation': 'Russia',
        'Viet Nam': 'Vietnam',
        "Lao People's Democratic Republic": 'Laos',
        'Syrian Arab Republic': 'Syria',
        'Czechia': 'Czech Republic',
        'Bahamas (the)': 'Bahamas',
        'Gambia (the)': 'Gambia',
        'Philippines (the)': 'Philippines',
        'Sudan (the)': 'Sudan',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'United States of America': 'United States',
        'Tanzania, the United Republic of': 'Tanzania',
        'Congo (the Democratic Republic of the)': 'DR Congo',
        'Congo (the)': 'Congo',
    }

    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing institutional datasets.
                     Defaults to RATCHET/data/institutional/
        """
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR

        # Raw data
        self._qog_data: Optional[pd.DataFrame] = None
        self._polity_data: Optional[pd.DataFrame] = None

        # Merged and processed data
        self._merged_data: Optional[pd.DataFrame] = None

        # Cached trajectories
        self._trajectories: Dict[str, CountryTrajectory] = {}

        # Collapse events
        self._collapse_events: List[CollapseEvent] = []

    def load_qog(self) -> pd.DataFrame:
        """Load QoG Standard Time-Series dataset."""
        qog_path = self.data_dir / self.QOG_FILENAME
        if not qog_path.exists():
            raise FileNotFoundError(
                f"QoG dataset not found at {qog_path}. "
                f"Download from https://www.qogdata.pol.gu.se/data/qog_std_ts_jan25.csv"
            )

        # Load with relevant columns only
        usecols = ['cname', 'ccodealp', 'year'] + list(self.QOG_VARS.keys())
        available_cols = pd.read_csv(qog_path, nrows=0).columns.tolist()
        usecols = [c for c in usecols if c in available_cols]

        self._qog_data = pd.read_csv(qog_path, usecols=usecols, low_memory=False)

        # Standardize country names
        self._qog_data['country'] = self._qog_data['cname'].map(
            lambda x: self.COUNTRY_NAME_MAP.get(x, x)
        )
        self._qog_data['country_code'] = self._qog_data['ccodealp']

        return self._qog_data

    def load_polity(self) -> pd.DataFrame:
        """Load Polity V dataset."""
        polity_path = self.data_dir / self.POLITY_FILENAME
        if not polity_path.exists():
            raise FileNotFoundError(
                f"Polity dataset not found at {polity_path}. "
                f"Download from http://www.systemicpeace.org/inscr/p5v2018.xls "
                f"and convert to CSV."
            )

        # Load Polity data
        usecols = ['country', 'scode', 'year'] + list(self.POLITY_VARS.keys())
        self._polity_data = pd.read_csv(polity_path)

        # Rename for consistency
        self._polity_data['country_code'] = self._polity_data['scode']

        return self._polity_data

    def load_all(self) -> pd.DataFrame:
        """Load and merge all available datasets."""
        # Load individual datasets
        try:
            self.load_qog()
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        try:
            self.load_polity()
        except FileNotFoundError as e:
            print(f"Warning: {e}")

        # Merge datasets
        self._merge_datasets()

        # Detect collapse events
        self._detect_collapse_events()

        return self._merged_data

    def _merge_datasets(self) -> None:
        """Merge QoG and Polity datasets."""
        if self._qog_data is None and self._polity_data is None:
            raise RuntimeError("No datasets loaded. Load at least one dataset first.")

        if self._qog_data is not None and self._polity_data is not None:
            # Merge on country and year
            # QoG contains Polity data already, but Polity has regtrans codes
            polity_cols = ['country', 'year', 'xconst', 'regtrans', 'change', 'sf', 'durable']
            polity_cols = [c for c in polity_cols if c in self._polity_data.columns]

            self._merged_data = self._qog_data.merge(
                self._polity_data[polity_cols],
                on=['country', 'year'],
                how='left',
                suffixes=('', '_polity')
            )
        elif self._qog_data is not None:
            self._merged_data = self._qog_data.copy()
        else:
            self._merged_data = self._polity_data.copy()

        # Compute RATCHET variables
        self._compute_ratchet_variables()

    def _compute_ratchet_variables(self) -> None:
        """Compute normalized RATCHET framework variables."""
        df = self._merged_data

        # k (constraints): From Polity xconst (1-7) or derive from democracy indices
        if 'xconst' in df.columns:
            # Polity xconst: 1 (unlimited authority) to 7 (executive parity)
            # Map to 0-1 where 1 is maximum constraints
            df['k'] = (df['xconst'] - 1) / 6.0
            df['k'] = df['k'].clip(0, 1)
        elif 'vdem_libdem' in df.columns:
            # Use V-Dem liberal democracy as proxy (already 0-1)
            df['k'] = df['vdem_libdem']
        else:
            df['k'] = np.nan

        # rho (elite coupling): Inverse of constraints, or derive from corruption
        # High corruption + low constraints = high elite capture
        if 'vdem_corr' in df.columns and 'k' in df.columns:
            # V-Dem corruption is 0-1 where 1 is most corrupt
            # Elite coupling = corruption * (1 - constraints)
            df['rho'] = df['vdem_corr'] * (1 - df['k'].fillna(0.5))
            df['rho'] = df['rho'].clip(0, 1)
        elif 'k' in df.columns:
            # Simple inverse of constraints
            df['rho'] = 1 - df['k']
        else:
            df['rho'] = np.nan

        # sigma (political stability): From V-Dem polyarchy or WBGI stability
        if 'vdem_polyarchy' in df.columns:
            # V-Dem polyarchy is 0-1, use directly
            df['sigma'] = df['vdem_polyarchy']
        elif 'wbgi_pve' in df.columns:
            # WBGI Political Stability: typically -2.5 to 2.5
            # Normalize to 0-1
            df['sigma'] = (df['wbgi_pve'] + 2.5) / 5.0
            df['sigma'] = df['sigma'].clip(0, 1)
        elif 'p_polity2' in df.columns:
            # Polity2: -10 to 10, normalize to 0-1
            df['sigma'] = (df['p_polity2'] + 10) / 20.0
            df['sigma'] = df['sigma'].clip(0, 1)
        else:
            df['sigma'] = np.nan

        # f (corruption): From V-Dem or WBGI
        if 'vdem_corr' in df.columns:
            # V-Dem corruption is 0-1 where 1 is most corrupt
            df['f'] = df['vdem_corr']
        elif 'vdem_pubcorr' in df.columns:
            df['f'] = df['vdem_pubcorr']
        elif 'wbgi_cce' in df.columns:
            # WBGI Control of Corruption: -2.5 to 2.5, higher is less corrupt
            # Invert and normalize so 1 = most corrupt
            df['f'] = (-df['wbgi_cce'] + 2.5) / 5.0
            df['f'] = df['f'].clip(0, 1)
        else:
            df['f'] = np.nan

        # lambda (rule of law): From WBGI Rule of Law
        if 'wbgi_rle' in df.columns:
            # WBGI Rule of Law: -2.5 to 2.5, normalize to 0-1
            df['lambda_'] = (df['wbgi_rle'] + 2.5) / 5.0
            df['lambda_'] = df['lambda_'].clip(0, 1)
        elif 'icrg_qog' in df.columns:
            # ICRG Quality of Government: 0-1
            df['lambda_'] = df['icrg_qog']
        else:
            df['lambda_'] = np.nan

        self._merged_data = df

    def _detect_collapse_events(self) -> None:
        """Detect institutional collapse events from Polity data."""
        if self._merged_data is None:
            return

        df = self._merged_data
        events = []

        for _, row in df.iterrows():
            country = row.get('country', '')
            country_code = row.get('country_code', '')
            year = row.get('year', 0)

            # Check regtrans code
            regtrans = row.get('regtrans', np.nan)
            polity_change = row.get('change', np.nan)
            sf = row.get('sf', np.nan)
            polity2 = row.get('p_polity2', row.get('polity2', np.nan))

            collapse_type = None
            description = ""

            # State failure (sf = 1)
            if sf == 1:
                collapse_type = CollapseType.STATE_FAILURE
                description = "State failure event (Polity SF flag)"

            # Adverse regime transition (regtrans = -2)
            elif regtrans == -2:
                collapse_type = CollapseType.ADVERSE_REGIME_TRANSITION
                description = "Adverse regime transition (Polity regtrans=-2)"

            # Large negative polity change (democratic breakdown)
            elif not np.isnan(polity_change) and polity_change <= -6:
                collapse_type = CollapseType.DEMOCRATIC_BREAKDOWN
                description = f"Democratic breakdown (polity change={polity_change})"

            # Autocratization (regtrans = -1 or moderate negative change)
            elif regtrans == -1:
                collapse_type = CollapseType.AUTOCRATIZATION
                description = "Negative regime change (Polity regtrans=-1)"

            if collapse_type is not None:
                event = CollapseEvent(
                    country=country,
                    country_code=country_code,
                    year=int(year),
                    collapse_type=collapse_type,
                    polity_change=polity_change if not np.isnan(polity_change) else None,
                    polity_before=None,  # Would need lag calculation
                    polity_after=polity2 if not np.isnan(polity2) else None,
                    regtrans_code=regtrans if not np.isnan(regtrans) else None,
                    description=description,
                )
                events.append(event)

        self._collapse_events = events

    def get_country_trajectory(
        self,
        country: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        interpolate: bool = True,
    ) -> CountryTrajectory:
        """
        Get RATCHET trajectory for a specific country.

        Args:
            country: Country name (standardized or original)
            start_year: Start year (default: earliest available)
            end_year: End year (default: latest available)
            interpolate: Whether to interpolate missing values

        Returns:
            CountryTrajectory with RATCHET variables
        """
        if self._merged_data is None:
            raise RuntimeError("Data not loaded. Call load_all() first.")

        # Find country (try standardized name first)
        df = self._merged_data
        country_data = df[df['country'] == country]

        if len(country_data) == 0:
            # Try original name
            country_data = df[df['cname'] == country]

        if len(country_data) == 0:
            available = sorted(df['country'].unique())
            raise ValueError(
                f"Country '{country}' not found. Available countries: {available[:20]}..."
            )

        # Filter by year range
        if start_year is not None:
            country_data = country_data[country_data['year'] >= start_year]
        if end_year is not None:
            country_data = country_data[country_data['year'] <= end_year]

        country_data = country_data.sort_values('year')

        # Extract RATCHET variables
        years = country_data['year'].values
        k = country_data['k'].values.astype(float)
        rho = country_data['rho'].values.astype(float)
        sigma = country_data['sigma'].values.astype(float)
        f = country_data['f'].values.astype(float)
        lambda_ = country_data['lambda_'].values.astype(float)

        # Interpolate missing values if requested
        if interpolate:
            k = self._interpolate_series(k)
            rho = self._interpolate_series(rho)
            sigma = self._interpolate_series(sigma)
            f = self._interpolate_series(f)
            lambda_ = self._interpolate_series(lambda_)

        # Get collapse events for this country
        country_events = [
            e for e in self._collapse_events
            if e.country == country or e.country == country_data['cname'].iloc[0]
        ]

        # Filter events by year range
        if start_year is not None or end_year is not None:
            country_events = [
                e for e in country_events
                if (start_year is None or e.year >= start_year) and
                   (end_year is None or e.year <= end_year)
            ]

        country_code = country_data['country_code'].iloc[0]

        return CountryTrajectory(
            country=country,
            country_code=country_code,
            years=years,
            k=k,
            rho=rho,
            sigma=sigma,
            f=f,
            lambda_=lambda_,
            collapse_events=country_events,
            raw_data=country_data,
        )

    def _interpolate_series(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate missing values in a series."""
        if np.all(np.isnan(arr)):
            return arr

        # Simple linear interpolation
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            return arr

        indices = np.arange(len(arr))
        arr_interp = np.interp(indices, indices[mask], arr[mask])

        return arr_interp

    def get_collapse_events(
        self,
        country: Optional[str] = None,
        collapse_type: Optional[CollapseType] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[CollapseEvent]:
        """
        Get collapse events, optionally filtered.

        Args:
            country: Filter by country name
            collapse_type: Filter by collapse type
            start_year: Filter by minimum year
            end_year: Filter by maximum year

        Returns:
            List of CollapseEvent objects
        """
        events = self._collapse_events

        if country is not None:
            events = [e for e in events if e.country == country]
        if collapse_type is not None:
            events = [e for e in events if e.collapse_type == collapse_type]
        if start_year is not None:
            events = [e for e in events if e.year >= start_year]
        if end_year is not None:
            events = [e for e in events if e.year <= end_year]

        return sorted(events, key=lambda e: (e.year, e.country))

    def list_countries(self) -> List[str]:
        """List all available countries."""
        if self._merged_data is None:
            raise RuntimeError("Data not loaded. Call load_all() first.")
        return sorted(self._merged_data['country'].unique())

    def get_countries_with_collapse(
        self,
        collapse_type: Optional[CollapseType] = None,
    ) -> List[str]:
        """Get list of countries that experienced collapse events."""
        events = self.get_collapse_events(collapse_type=collapse_type)
        return sorted(set(e.country for e in events))

    def summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        if self._merged_data is None:
            return {'loaded': False}

        df = self._merged_data
        return {
            'loaded': True,
            'n_countries': df['country'].nunique(),
            'n_observations': len(df),
            'year_range': (int(df['year'].min()), int(df['year'].max())),
            'n_collapse_events': len(self._collapse_events),
            'collapse_types': {
                ct.value: sum(1 for e in self._collapse_events if e.collapse_type == ct)
                for ct in CollapseType
            },
            'variable_coverage': {
                'k': df['k'].notna().mean(),
                'rho': df['rho'].notna().mean(),
                'sigma': df['sigma'].notna().mean(),
                'f': df['f'].notna().mean(),
                'lambda_': df['lambda_'].notna().mean(),
            }
        }


__all__ = [
    'InstitutionalDataLoader',
    'CountryTrajectory',
    'CollapseEvent',
    'CollapseType',
]
