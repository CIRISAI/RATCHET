"""
RATCHET Federation Module

Provides BFT (Byzantine Fault Tolerant) backends and utilities for
distributed precedent accumulation.

Components:
- InMemoryTransport: Asyncio-based message passing for testing
- InMemoryStorage: Dict-based storage for logs and checkpoints
- Ed25519CryptoProvider: Ed25519 signing/verification

These backends implement the abstract interfaces from schemas/bft.py
and are used by the FederationEngine for PBFT consensus.
"""

from .backends import (
    InMemoryTransport,
    InMemoryStorage,
    Ed25519CryptoProvider,
)

__all__ = [
    'InMemoryTransport',
    'InMemoryStorage',
    'Ed25519CryptoProvider',
]
