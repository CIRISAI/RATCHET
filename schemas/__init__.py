"""
RATCHET Schema Module

Pydantic models for protocol messages, experiment configuration, and results.
"""

from .bft import (
    # Configuration
    BFTConfig,
    MessageType,
    ReplicaStatus,

    # Core message types
    Digest,
    Signature,
    Request,
    PrePrepare,
    Prepare,
    Commit,
    Reply,

    # View change protocol
    PreparedCertificate,
    CheckpointMessage,
    ViewChange,
    NewView,

    # Federation extensions
    PrecedentOperation,
    FederationState,

    # Utilities
    MessageLog,
    BFTMetrics,

    # Verification helpers
    verify_bft_invariant,
    compute_max_faulty,
    compute_min_replicas,
    compute_quorum_size,

    # Type aliases
    ReplicaId,
    ViewNumber,
    SequenceNumber,
    ClientId,
    PBFTMessage,
)

__all__ = [
    # Configuration
    "BFTConfig",
    "MessageType",
    "ReplicaStatus",

    # Core message types
    "Digest",
    "Signature",
    "Request",
    "PrePrepare",
    "Prepare",
    "Commit",
    "Reply",

    # View change protocol
    "PreparedCertificate",
    "CheckpointMessage",
    "ViewChange",
    "NewView",

    # Federation extensions
    "PrecedentOperation",
    "FederationState",

    # Utilities
    "MessageLog",
    "BFTMetrics",

    # Verification helpers
    "verify_bft_invariant",
    "compute_max_faulty",
    "compute_min_replicas",
    "compute_quorum_size",

    # Type aliases
    "ReplicaId",
    "ViewNumber",
    "SequenceNumber",
    "ClientId",
    "PBFTMessage",
]
