"""
Core EPM (Error Patch Memory) components
"""

from .epm_schema import (
    ERROR_FAMILIES,
    FAILURE_CATEGORIES,
    LEGACY_TO_MAST_MAPPING,
    ErrorSignature,
    ErrorSpan,
    ErrorPatch,
    Evidence,
    RFIScores,
    EPMMetadata,
    EPMEntry,
    ErrorHypothesis,
    EPMConfig,
    get_failure_category,
    migrate_legacy_family,
    is_mast_family,
    get_all_mast_families,
    get_family_statistics
)

from .epm_manager import EPMManager

__all__ = [
    'ERROR_FAMILIES',
    'FAILURE_CATEGORIES',
    'LEGACY_TO_MAST_MAPPING',
    'ErrorSignature',
    'ErrorSpan',
    'ErrorPatch',
    'Evidence',
    'RFIScores',
    'EPMMetadata',
    'EPMEntry',
    'ErrorHypothesis',
    'EPMConfig',
    'EPMManager',
    'get_failure_category',
    'migrate_legacy_family',
    'is_mast_family',
    'get_all_mast_families',
    'get_family_statistics'
]
