"""
RATCHET Data Loaders

Provides data loading and preprocessing for empirical datasets used with RATCHET engines.
"""

from __future__ import annotations

__all__ = [
    # Institutional data
    'InstitutionalDataLoader',
    'CountryTrajectory',
    'CollapseEvent',
    # Microbiome data
    'MicrobiomeDataLoader',
    'MicrobiomeSample',
    'SyntheticMicrobiomeGenerator',
    'load_american_gut_project',
    # Battery data
    'NASABatteryLoader',
    'NASABatteryDataset',
    'BatteryData',
    'BatteryCycleData',
    'load_nasa_battery_data',
    'get_high_quality_cells',
    'save_processed_data',
    'prepare_for_engine',
    # Ecological / BioTIME data
    'EcologicalSample',
    'BioTIMECommunityDataset',
    'SyntheticBioTIMEGenerator',
    'load_biotime_data',
    'load_biotime_communities',
]

# Lazy imports to avoid heavy dependencies at module load time
def __getattr__(name: str):
    if name in ('InstitutionalDataLoader', 'CountryTrajectory', 'CollapseEvent'):
        from ratchet.data.institutional_loader import (
            InstitutionalDataLoader,
            CountryTrajectory,
            CollapseEvent,
        )
        return locals()[name]
    if name in ('MicrobiomeDataLoader', 'MicrobiomeSample', 'SyntheticMicrobiomeGenerator', 'load_american_gut_project'):
        from ratchet.data.microbiome_loader import (
            MicrobiomeDataLoader,
            MicrobiomeSample,
            SyntheticMicrobiomeGenerator,
            load_american_gut_project,
        )
        return locals()[name]
    if name in ('NASABatteryLoader', 'NASABatteryDataset', 'BatteryData', 'BatteryCycleData',
                'load_nasa_battery_data', 'get_high_quality_cells', 'save_processed_data', 'prepare_for_engine'):
        from ratchet.data.battery_loader import (
            NASABatteryLoader,
            NASABatteryDataset,
            BatteryData,
            BatteryCycleData,
            load_nasa_battery_data,
            get_high_quality_cells,
            save_processed_data,
            prepare_for_engine,
        )
        return locals()[name]
    if name in ('EcologicalSample', 'BioTIMECommunityDataset', 'SyntheticBioTIMEGenerator',
                'load_biotime_data', 'load_biotime_communities'):
        from ratchet.data.ecological_loader import (
            EcologicalSample,
            BioTIMECommunityDataset,
            SyntheticBioTIMEGenerator,
            load_biotime_data,
            load_biotime_communities,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
