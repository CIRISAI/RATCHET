"""
PBFT (Practical Byzantine Fault Tolerance) Message Types for RATCHET Federation

This module defines the complete message format specification for PBFT consensus
used in the federation engine. Based on Castro & Liskov's PBFT protocol (1999).

Byzantine Fault Tolerance Invariant:
    n >= 3f + 1

    Where:
    - n = total number of replicas (federation nodes)
    - f = maximum number of Byzantine (faulty/malicious) nodes tolerated

    This ensures safety and liveness with up to f Byzantine failures.

Protocol Phases:
    1. REQUEST: Client sends operation to primary
    2. PRE-PREPARE: Primary assigns sequence number, broadcasts to replicas
    3. PREPARE: Replicas validate and broadcast prepare messages
    4. COMMIT: After 2f+1 prepares, replicas broadcast commit
    5. REPLY: After 2f+1 commits, replicas execute and reply to client

View Change Protocol:
    When primary fails or is suspected Byzantine, replicas initiate view change
    to elect new primary. Requires 2f+1 VIEW-CHANGE messages to proceed.
"""

from __future__ import annotations

import hashlib
import time
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

class BFTConfig(BaseModel):
    """Configuration for PBFT consensus protocol."""

    # Timeout parameters (in milliseconds)
    request_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Timeout for client request before retry"
    )
    preprepare_timeout_ms: int = Field(
        default=2000,
        ge=100,
        le=30000,
        description="Timeout waiting for PRE-PREPARE from primary"
    )
    prepare_timeout_ms: int = Field(
        default=3000,
        ge=100,
        le=30000,
        description="Timeout waiting for 2f+1 PREPARE messages"
    )
    commit_timeout_ms: int = Field(
        default=3000,
        ge=100,
        le=30000,
        description="Timeout waiting for 2f+1 COMMIT messages"
    )
    view_change_timeout_ms: int = Field(
        default=10000,
        ge=1000,
        le=120000,
        description="Timeout before initiating view change"
    )
    checkpoint_interval: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of requests between stable checkpoints"
    )

    # Water marks for garbage collection
    low_water_mark_window: int = Field(
        default=200,
        ge=100,
        le=10000,
        description="h - low water mark (sequence numbers below are garbage collected)"
    )
    high_water_mark_window: int = Field(
        default=400,
        ge=200,
        le=20000,
        description="H - high water mark (reject requests with seq > H)"
    )

    @model_validator(mode='after')
    def validate_water_marks(self) -> 'BFTConfig':
        if self.high_water_mark_window <= self.low_water_mark_window:
            raise ValueError("high_water_mark_window must be > low_water_mark_window")
        return self


class MessageType(str, Enum):
    """PBFT message types."""
    REQUEST = "REQUEST"
    PRE_PREPARE = "PRE-PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    REPLY = "REPLY"
    VIEW_CHANGE = "VIEW-CHANGE"
    NEW_VIEW = "NEW-VIEW"
    CHECKPOINT = "CHECKPOINT"


class ReplicaStatus(str, Enum):
    """Status of a replica node."""
    ACTIVE = "ACTIVE"
    SUSPECTED = "SUSPECTED"
    FAULTY = "FAULTY"
    VIEW_CHANGING = "VIEW_CHANGING"


# =============================================================================
# CORE MESSAGE TYPES
# =============================================================================

class Digest(BaseModel):
    """Cryptographic digest of a message for verification."""

    algorithm: str = Field(default="sha256", description="Hash algorithm used")
    value: str = Field(..., min_length=64, max_length=64, description="Hex-encoded hash")

    @classmethod
    def compute(cls, data: bytes) -> 'Digest':
        """Compute digest of data."""
        hash_value = hashlib.sha256(data).hexdigest()
        return cls(algorithm="sha256", value=hash_value)

    @classmethod
    def from_message(cls, message: BaseModel) -> 'Digest':
        """Compute digest from a Pydantic model."""
        # Serialize deterministically
        data = message.model_dump_json(exclude={'digest', 'signature'}).encode('utf-8')
        return cls.compute(data)


class Signature(BaseModel):
    """Digital signature for message authentication."""

    signer_id: str = Field(..., description="ID of the signing replica")
    algorithm: str = Field(default="ed25519", description="Signature algorithm")
    value: str = Field(..., description="Base64-encoded signature")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Signing timestamp in milliseconds"
    )


class Request(BaseModel):
    """
    Client request message (Phase 1).

    Clients send requests to the primary replica. If no response within
    request_timeout_ms, client broadcasts to all replicas.

    Format: <REQUEST, o, t, c>_sigma_c
    - o: operation to execute
    - t: timestamp (for exactly-once semantics)
    - c: client identifier
    """

    message_type: MessageType = Field(default=MessageType.REQUEST, frozen=True)
    operation: Dict[str, Any] = Field(..., description="Operation to execute")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Client timestamp for exactly-once semantics"
    )
    client_id: str = Field(..., min_length=1, description="Client identifier")
    request_id: UUID = Field(default_factory=uuid4, description="Unique request ID")
    signature: Optional[Signature] = Field(default=None, description="Client signature")

    def digest(self) -> Digest:
        """Compute digest of this request."""
        return Digest.from_message(self)


class PrePrepare(BaseModel):
    """
    Pre-prepare message (Phase 2) - sent by primary only.

    Primary assigns sequence number and broadcasts to all replicas.
    Replicas accept if:
    1. Signatures verify
    2. View number matches current view
    3. Sequence number is between water marks (h < n <= H)
    4. No other pre-prepare for same view/sequence with different digest

    Format: <<PRE-PREPARE, v, n, d>_sigma_p, m>
    - v: view number
    - n: sequence number
    - d: digest of request m
    """

    message_type: MessageType = Field(default=MessageType.PRE_PREPARE, frozen=True)
    view_number: int = Field(..., ge=0, description="Current view number")
    sequence_number: int = Field(..., ge=0, description="Assigned sequence number")
    request_digest: Digest = Field(..., description="Digest of client request")
    request: Request = Field(..., description="Original client request")
    primary_id: str = Field(..., description="ID of primary replica")
    signature: Optional[Signature] = Field(default=None, description="Primary signature")

    @field_validator('view_number', 'sequence_number')
    @classmethod
    def validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Must be non-negative")
        return v


class Prepare(BaseModel):
    """
    Prepare message (Phase 3) - broadcast by replicas.

    Replica i multicasts PREPARE to all other replicas if it accepts PRE-PREPARE.
    Prepared(m, v, n, i) is true when replica i has:
    1. The request m
    2. A PRE-PREPARE for m in view v with sequence n
    3. 2f matching PREPARE messages from different replicas

    Format: <PREPARE, v, n, d, i>_sigma_i
    - v: view number
    - n: sequence number
    - d: digest
    - i: replica identifier
    """

    message_type: MessageType = Field(default=MessageType.PREPARE, frozen=True)
    view_number: int = Field(..., ge=0, description="Current view number")
    sequence_number: int = Field(..., ge=0, description="Sequence number from PRE-PREPARE")
    request_digest: Digest = Field(..., description="Digest of client request")
    replica_id: str = Field(..., description="ID of sending replica")
    signature: Optional[Signature] = Field(default=None, description="Replica signature")


class Commit(BaseModel):
    """
    Commit message (Phase 4) - broadcast after prepared.

    Replica i multicasts COMMIT after prepared(m, v, n, i) is true.
    Committed-local(m, v, n, i) is true when:
    1. prepared(m, v, n, i) is true
    2. Replica has 2f+1 matching COMMIT messages from different replicas

    Format: <COMMIT, v, n, D(m), i>_sigma_i
    """

    message_type: MessageType = Field(default=MessageType.COMMIT, frozen=True)
    view_number: int = Field(..., ge=0, description="Current view number")
    sequence_number: int = Field(..., ge=0, description="Sequence number")
    request_digest: Digest = Field(..., description="Digest of client request")
    replica_id: str = Field(..., description="ID of sending replica")
    signature: Optional[Signature] = Field(default=None, description="Replica signature")


class Reply(BaseModel):
    """
    Reply message (Phase 5) - sent to client.

    After committed-local, replica executes operation and sends reply.
    Client waits for f+1 matching replies with same result.

    Format: <REPLY, v, t, c, i, r>_sigma_i
    - v: current view
    - t: timestamp from request
    - c: client id
    - i: replica id
    - r: result of operation
    """

    message_type: MessageType = Field(default=MessageType.REPLY, frozen=True)
    view_number: int = Field(..., ge=0, description="View in which request executed")
    request_timestamp_ms: int = Field(..., description="Timestamp from original request")
    client_id: str = Field(..., description="Client that sent request")
    replica_id: str = Field(..., description="ID of replying replica")
    result: Dict[str, Any] = Field(..., description="Execution result")
    request_id: UUID = Field(..., description="Original request ID")
    signature: Optional[Signature] = Field(default=None, description="Replica signature")


# =============================================================================
# VIEW CHANGE PROTOCOL
# =============================================================================

class PreparedCertificate(BaseModel):
    """
    Certificate proving a request was prepared.

    Contains PRE-PREPARE and 2f matching PREPARE messages.
    Used in VIEW-CHANGE to prove what was prepared in previous views.
    """

    pre_prepare: PrePrepare
    prepares: List[Prepare] = Field(..., min_length=1)

    @model_validator(mode='after')
    def validate_certificate(self) -> 'PreparedCertificate':
        """Ensure all prepares match the pre-prepare."""
        pp = self.pre_prepare
        for prepare in self.prepares:
            if prepare.view_number != pp.view_number:
                raise ValueError("PREPARE view must match PRE-PREPARE view")
            if prepare.sequence_number != pp.sequence_number:
                raise ValueError("PREPARE sequence must match PRE-PREPARE sequence")
            if prepare.request_digest.value != pp.request_digest.value:
                raise ValueError("PREPARE digest must match PRE-PREPARE digest")
        return self


class CheckpointMessage(BaseModel):
    """
    Checkpoint message for garbage collection.

    Replicas periodically checkpoint state. A checkpoint is stable when
    2f+1 replicas have matching checkpoints.

    Format: <CHECKPOINT, n, d, i>_sigma_i
    - n: sequence number of last executed request
    - d: digest of state
    """

    message_type: MessageType = Field(default=MessageType.CHECKPOINT, frozen=True)
    sequence_number: int = Field(..., ge=0, description="Last executed sequence number")
    state_digest: Digest = Field(..., description="Digest of application state")
    replica_id: str = Field(..., description="ID of sending replica")
    signature: Optional[Signature] = Field(default=None, description="Replica signature")


class ViewChange(BaseModel):
    """
    View change message - initiates leader election.

    Sent when replica suspects primary is faulty. Contains proof of
    what was prepared in previous views so new primary can resume.

    Format: <VIEW-CHANGE, v+1, n, C, P, i>_sigma_i
    - v+1: new view number
    - n: sequence number of last stable checkpoint
    - C: 2f+1 checkpoint messages proving stable checkpoint
    - P: set of prepared certificates for requests after checkpoint
    """

    message_type: MessageType = Field(default=MessageType.VIEW_CHANGE, frozen=True)
    new_view_number: int = Field(..., ge=1, description="Proposed new view number")
    last_stable_checkpoint: int = Field(..., ge=0, description="Sequence of stable checkpoint")
    checkpoint_proofs: List[CheckpointMessage] = Field(
        default_factory=list,
        description="2f+1 checkpoint messages for stable checkpoint"
    )
    prepared_certificates: List[PreparedCertificate] = Field(
        default_factory=list,
        description="Certificates for prepared but not committed requests"
    )
    replica_id: str = Field(..., description="ID of replica initiating view change")
    signature: Optional[Signature] = Field(default=None, description="Replica signature")

    @field_validator('new_view_number')
    @classmethod
    def validate_new_view(cls, v: int) -> int:
        if v < 1:
            raise ValueError("new_view_number must be at least 1")
        return v


class NewView(BaseModel):
    """
    New view message - sent by new primary to complete view change.

    New primary collects 2f+1 VIEW-CHANGE messages, computes which
    requests must be re-proposed, and sends NEW-VIEW.

    Format: <NEW-VIEW, v+1, V, O>_sigma_p
    - v+1: new view number
    - V: set of 2f+1 valid VIEW-CHANGE messages
    - O: set of PRE-PREPARE messages for requests to re-propose
    """

    message_type: MessageType = Field(default=MessageType.NEW_VIEW, frozen=True)
    new_view_number: int = Field(..., ge=1, description="New view number")
    view_change_proofs: List[ViewChange] = Field(
        ...,
        min_length=1,
        description="2f+1 VIEW-CHANGE messages"
    )
    pre_prepares_to_redo: List[PrePrepare] = Field(
        default_factory=list,
        description="PRE-PREPAREs for requests that must be re-proposed"
    )
    new_primary_id: str = Field(..., description="ID of new primary")
    signature: Optional[Signature] = Field(default=None, description="New primary signature")


# =============================================================================
# FEDERATION-SPECIFIC EXTENSIONS
# =============================================================================

class PrecedentOperation(BaseModel):
    """
    Operation type for federation precedent voting.

    Used as the 'operation' field in Request messages for
    precedent-related consensus.
    """

    operation_type: str = Field(..., description="Type: 'ADD_PRECEDENT', 'VOTE', 'CHALLENGE'")
    precedent_id: Optional[UUID] = Field(default=None, description="ID of precedent")
    precedent_content: Optional[str] = Field(default=None, description="Precedent statement")
    vote: Optional[bool] = Field(default=None, description="Vote on precedent (True=accept)")
    challenge_reason: Optional[str] = Field(default=None, description="Reason for challenge")
    submitter_id: str = Field(..., description="ID of submitting agent")


class FederationState(BaseModel):
    """
    State of the federation at a checkpoint.

    Contains accepted precedents, member list, and integrity data.
    """

    view_number: int = Field(..., ge=0, description="Current view")
    last_sequence: int = Field(..., ge=0, description="Last executed sequence")

    # Federation membership
    total_replicas: int = Field(..., ge=1, description="n - total replicas")
    max_faulty: int = Field(..., ge=0, description="f - max Byzantine nodes")
    replica_ids: List[str] = Field(..., description="IDs of all replicas")
    primary_id: str = Field(..., description="Current primary")

    # Precedent state
    accepted_precedents: List[UUID] = Field(
        default_factory=list,
        description="IDs of accepted precedents"
    )
    pending_precedents: List[UUID] = Field(
        default_factory=list,
        description="IDs of pending precedents"
    )

    # Integrity
    state_digest: Optional[Digest] = Field(default=None, description="State hash")
    timestamp_ms: int = Field(
        default_factory=lambda: int(time.time() * 1000),
        description="Checkpoint timestamp"
    )

    @model_validator(mode='after')
    def validate_bft_invariant(self) -> 'FederationState':
        """
        Verify Byzantine fault tolerance invariant: n >= 3f + 1

        This is the CRITICAL safety invariant for PBFT. With n replicas
        and f Byzantine nodes:
        - Need 2f+1 for quorum (majority of honest nodes)
        - Need n - f >= 2f + 1 honest nodes to ensure quorum is honest
        - Therefore n >= 3f + 1
        """
        n = self.total_replicas
        f = self.max_faulty

        if n < 3 * f + 1:
            raise ValueError(
                f"BFT invariant violated: n={n} must be >= 3f+1={3*f+1} "
                f"to tolerate f={f} Byzantine failures"
            )

        if len(self.replica_ids) != n:
            raise ValueError(
                f"Replica ID list length ({len(self.replica_ids)}) must equal n ({n})"
            )

        if self.primary_id not in self.replica_ids:
            raise ValueError("Primary must be in replica list")

        return self

    def quorum_size(self) -> int:
        """Return required quorum size (2f + 1)."""
        return 2 * self.max_faulty + 1

    def is_quorum(self, count: int) -> bool:
        """Check if count meets quorum threshold."""
        return count >= self.quorum_size()

    def reply_threshold(self) -> int:
        """Return f + 1 for client reply acceptance."""
        return self.max_faulty + 1


# =============================================================================
# UTILITY TYPES
# =============================================================================

class MessageLog(BaseModel):
    """
    Log of messages for a specific sequence number.

    Tracks progress through PBFT phases.
    """

    sequence_number: int = Field(..., ge=0)
    view_number: int = Field(..., ge=0)

    request: Optional[Request] = None
    pre_prepare: Optional[PrePrepare] = None
    prepares: Dict[str, Prepare] = Field(default_factory=dict)  # replica_id -> Prepare
    commits: Dict[str, Commit] = Field(default_factory=dict)  # replica_id -> Commit

    prepared: bool = False
    committed_local: bool = False
    executed: bool = False

    def prepare_count(self) -> int:
        return len(self.prepares)

    def commit_count(self) -> int:
        return len(self.commits)


class BFTMetrics(BaseModel):
    """Metrics for PBFT protocol health."""

    current_view: int = Field(..., ge=0)
    last_executed_sequence: int = Field(..., ge=0)
    pending_requests: int = Field(..., ge=0)

    # Timing metrics (in milliseconds)
    avg_consensus_latency_ms: float = Field(..., ge=0)
    avg_view_change_duration_ms: float = Field(..., ge=0)

    # Health indicators
    view_changes_last_hour: int = Field(..., ge=0)
    suspected_replicas: List[str] = Field(default_factory=list)

    # Throughput
    requests_per_second: float = Field(..., ge=0)
    messages_per_consensus: float = Field(..., ge=0)


# =============================================================================
# VERIFICATION HELPERS
# =============================================================================

def verify_bft_invariant(n: int, f: int) -> bool:
    """
    Verify the Byzantine Fault Tolerance invariant.

    Args:
        n: Total number of replicas
        f: Maximum number of Byzantine (faulty) replicas

    Returns:
        True if n >= 3f + 1, False otherwise

    The invariant n >= 3f + 1 ensures:
    - Safety: Faulty replicas cannot cause disagreement
    - Liveness: System makes progress with up to f failures

    Example:
        >>> verify_bft_invariant(4, 1)  # 4 >= 3(1) + 1 = 4
        True
        >>> verify_bft_invariant(3, 1)  # 3 >= 3(1) + 1 = 4
        False
        >>> verify_bft_invariant(7, 2)  # 7 >= 3(2) + 1 = 7
        True
    """
    return n >= 3 * f + 1


def compute_max_faulty(n: int) -> int:
    """
    Compute maximum tolerable Byzantine nodes for given replica count.

    Args:
        n: Total number of replicas

    Returns:
        f = floor((n - 1) / 3)

    Example:
        >>> compute_max_faulty(4)
        1
        >>> compute_max_faulty(7)
        2
        >>> compute_max_faulty(10)
        3
    """
    return (n - 1) // 3


def compute_min_replicas(f: int) -> int:
    """
    Compute minimum replicas needed to tolerate f Byzantine failures.

    Args:
        f: Desired Byzantine fault tolerance

    Returns:
        n = 3f + 1

    Example:
        >>> compute_min_replicas(1)
        4
        >>> compute_min_replicas(2)
        7
        >>> compute_min_replicas(3)
        10
    """
    return 3 * f + 1


def compute_quorum_size(n: int) -> int:
    """
    Compute quorum size for n replicas.

    Quorum = 2f + 1 where f = floor((n-1)/3)

    Args:
        n: Total number of replicas

    Returns:
        Quorum size ensuring overlap between any two quorums contains
        at least one honest replica.
    """
    f = compute_max_faulty(n)
    return 2 * f + 1


# Type aliases for clarity
ReplicaId = str
ViewNumber = int
SequenceNumber = int
ClientId = str

# Union of all PBFT message types
PBFTMessage = Union[
    Request,
    PrePrepare,
    Prepare,
    Commit,
    Reply,
    ViewChange,
    NewView,
    CheckpointMessage,
]
