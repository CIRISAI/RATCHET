"""
RATCHET Federation Engine

Implements PBFT consensus for distributed precedent accumulation with:
- Byzantine Fault Tolerance (n >= 3f + 1)
- MI threshold gate for partnership qualification
- Behavioral correlation detection for Sybil resistance
- Slow capture attack detection

Uses FederationParams from schemas/simulation.py and BFT types from schemas/bft.py.

Key invariant: The federation maintains safety as long as malicious_fraction < 1/3.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas.bft import (
    BFTConfig,
    CheckpointMessage,
    Commit,
    Digest,
    FederationState,
    MessageLog,
    MessageType,
    NewView,
    PBFTMessage,
    PrePrepare,
    Prepare,
    Reply,
    Request,
    Signature,
    ViewChange,
    compute_max_faulty,
    compute_quorum_size,
    verify_bft_invariant,
)
from schemas.simulation import FederationParams
from schemas.types import (
    ConsensusProtocol,
    MaliciousStrategy,
    NodeCount,
    Precedent,
    Vote,
)

from ratchet.federation.backends import (
    Ed25519CryptoProvider,
    InMemoryStorage,
    InMemoryTransport,
)


# =============================================================================
# NODE TYPES
# =============================================================================

class NodeType(str, Enum):
    """Classification of federation nodes."""
    HONEST = "honest"
    MALICIOUS = "malicious"


@dataclass
class FederationNode:
    """
    Represents a node in the federation.

    Each node has:
    - Unique identifier
    - Cryptographic identity (Ed25519 keypair)
    - Behavioral trace for correlation detection
    - Partnership status
    """
    node_id: str
    node_type: NodeType
    crypto: Ed25519CryptoProvider
    transport: InMemoryTransport
    storage: InMemoryStorage

    # Behavioral trace for Sybil detection
    behavioral_trace: List[float] = field(default_factory=list)

    # MI value with federation (for partnership gate)
    mi_value: float = 0.0

    # Partnership status
    is_partner: bool = False

    # For malicious nodes: their strategy
    malicious_strategy: Optional[MaliciousStrategy] = None

    def get_public_key(self) -> bytes:
        """Return this node's public key."""
        return self.crypto.get_public_key()


@dataclass
class ConsensusRound:
    """
    State of a single consensus round.

    Tracks progress through PBFT phases for one request.
    """
    sequence_number: int
    view_number: int
    request: Request
    request_digest: Digest

    # Phase tracking
    pre_prepare: Optional[PrePrepare] = None
    prepares: Dict[str, Prepare] = field(default_factory=dict)
    commits: Dict[str, Commit] = field(default_factory=dict)
    replies: Dict[str, Reply] = field(default_factory=dict)

    # Status flags
    prepared: bool = False
    committed_local: bool = False
    executed: bool = False

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# =============================================================================
# SYBIL DETECTION
# =============================================================================

class BehavioralCorrelationDetector:
    """
    Detects Sybil attacks through behavioral correlation analysis.

    Sybil nodes controlled by the same adversary tend to exhibit
    correlated behavior patterns. This detector identifies such
    correlations using:

    1. Behavioral trace collection (voting patterns, timing, etc.)
    2. Pairwise correlation computation
    3. Clustering of highly correlated nodes
    4. Flagging of suspicious clusters

    Theoretical basis: Independent honest nodes have low correlation,
    while Sybil nodes share information and thus correlate highly.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.8,
        min_trace_length: int = 10,
    ):
        """
        Initialize detector.

        Args:
            correlation_threshold: Threshold above which nodes are flagged
            min_trace_length: Minimum trace length before detection is valid
        """
        self.correlation_threshold = correlation_threshold
        self.min_trace_length = min_trace_length

    def compute_correlation(
        self,
        trace1: List[float],
        trace2: List[float],
    ) -> float:
        """
        Compute Pearson correlation between two behavioral traces.

        Args:
            trace1: First behavioral trace
            trace2: Second behavioral trace

        Returns:
            Correlation coefficient in [-1, 1]
        """
        if len(trace1) < self.min_trace_length or len(trace2) < self.min_trace_length:
            return 0.0

        # Align traces to same length
        min_len = min(len(trace1), len(trace2))
        t1 = np.array(trace1[-min_len:])
        t2 = np.array(trace2[-min_len:])

        # Compute Pearson correlation
        if np.std(t1) < 1e-10 or np.std(t2) < 1e-10:
            return 0.0

        correlation = np.corrcoef(t1, t2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0

    def detect_sybil_clusters(
        self,
        nodes: List[FederationNode],
    ) -> List[Set[str]]:
        """
        Detect clusters of potentially Sybil nodes.

        Args:
            nodes: List of federation nodes to analyze

        Returns:
            List of sets, each set containing node IDs that appear to be Sybils
        """
        # Build correlation matrix
        n = len(nodes)
        correlations: Dict[Tuple[str, str], float] = {}

        for i in range(n):
            for j in range(i + 1, n):
                node_i = nodes[i]
                node_j = nodes[j]
                corr = self.compute_correlation(
                    node_i.behavioral_trace,
                    node_j.behavioral_trace
                )
                correlations[(node_i.node_id, node_j.node_id)] = corr

        # Find connected components above threshold
        suspicious_pairs: List[Tuple[str, str]] = [
            pair for pair, corr in correlations.items()
            if corr > self.correlation_threshold
        ]

        # Build clusters using union-find
        clusters: Dict[str, Set[str]] = {}
        for id1, id2 in suspicious_pairs:
            # Find existing clusters
            cluster1 = None
            cluster2 = None
            for root, members in clusters.items():
                if id1 in members:
                    cluster1 = root
                if id2 in members:
                    cluster2 = root

            if cluster1 is None and cluster2 is None:
                # New cluster
                clusters[id1] = {id1, id2}
            elif cluster1 is not None and cluster2 is None:
                clusters[cluster1].add(id2)
            elif cluster1 is None and cluster2 is not None:
                clusters[cluster2].add(id1)
            elif cluster1 != cluster2:
                # Merge clusters
                clusters[cluster1].update(clusters[cluster2])
                del clusters[cluster2]

        # Return clusters with more than 1 member (suspicious)
        return [members for members in clusters.values() if len(members) > 1]

    def is_suspicious(
        self,
        node: FederationNode,
        other_nodes: List[FederationNode],
    ) -> Tuple[bool, float]:
        """
        Check if a single node shows suspicious correlation with others.

        Args:
            node: Node to check
            other_nodes: Other nodes to compare against

        Returns:
            Tuple of (is_suspicious, max_correlation)
        """
        max_corr = 0.0
        for other in other_nodes:
            if other.node_id != node.node_id:
                corr = self.compute_correlation(
                    node.behavioral_trace,
                    other.behavioral_trace
                )
                max_corr = max(max_corr, abs(corr))

        return (max_corr > self.correlation_threshold, max_corr)


# =============================================================================
# MI THRESHOLD GATE
# =============================================================================

class MIThresholdGate:
    """
    Mutual Information threshold gate for partnership qualification.

    Nodes must demonstrate sufficient information-theoretic independence
    (low MI with existing partners) to qualify for partnership.

    This prevents:
    1. Sybil attacks (correlated nodes would have high MI)
    2. Echo chambers (redundant information contributors)
    3. Collusion (coordinated malicious nodes)
    """

    def __init__(
        self,
        mi_threshold: float = 0.85,
        min_observations: int = 10,
    ):
        """
        Initialize MI gate.

        Args:
            mi_threshold: Maximum allowed MI for partnership (0 to 1)
            min_observations: Minimum observations before gate applies
        """
        self.mi_threshold = mi_threshold
        self.min_observations = min_observations

    def compute_mi(
        self,
        trace1: List[float],
        trace2: List[float],
        bins: int = 10,
    ) -> float:
        """
        Estimate mutual information between two behavioral traces.

        Uses binned histogram estimation of MI:
        I(X;Y) = H(X) + H(Y) - H(X,Y)

        Args:
            trace1: First behavioral trace
            trace2: Second behavioral trace
            bins: Number of bins for histogram

        Returns:
            Estimated MI (normalized to [0, 1])
        """
        if len(trace1) < self.min_observations or len(trace2) < self.min_observations:
            return 0.0

        # Align traces
        min_len = min(len(trace1), len(trace2))
        t1 = np.array(trace1[-min_len:])
        t2 = np.array(trace2[-min_len:])

        # Compute 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(t1, t2, bins=bins)
        hist_2d = hist_2d / hist_2d.sum()  # Normalize

        # Marginal distributions
        px = hist_2d.sum(axis=1)
        py = hist_2d.sum(axis=0)

        # Compute entropies
        eps = 1e-10
        h_x = -np.sum(px * np.log2(px + eps))
        h_y = -np.sum(py * np.log2(py + eps))
        h_xy = -np.sum(hist_2d * np.log2(hist_2d + eps))

        # Mutual information
        mi = h_x + h_y - h_xy

        # Normalize by max possible MI
        max_mi = min(h_x, h_y)
        if max_mi < eps:
            return 0.0

        return min(1.0, mi / max_mi)

    def qualifies_for_partnership(
        self,
        candidate: FederationNode,
        existing_partners: List[FederationNode],
    ) -> Tuple[bool, float]:
        """
        Check if a candidate qualifies for partnership.

        Args:
            candidate: Node seeking partnership
            existing_partners: Current partner nodes

        Returns:
            Tuple of (qualifies, avg_mi)
        """
        if not existing_partners:
            return (True, 0.0)

        mi_values = []
        for partner in existing_partners:
            mi = self.compute_mi(
                candidate.behavioral_trace,
                partner.behavioral_trace
            )
            mi_values.append(mi)

        avg_mi = np.mean(mi_values)
        max_mi = np.max(mi_values)

        # Must be below threshold with ALL partners
        qualifies = max_mi < self.mi_threshold

        return (qualifies, float(avg_mi))


# =============================================================================
# FEDERATION ENGINE
# =============================================================================

class FederationEngine:
    """
    Federation Engine implementing PBFT consensus.

    Manages a federation of nodes that achieve Byzantine Fault Tolerant
    consensus on precedents. Implements:

    1. PBFT consensus protocol (Castro & Liskov, 1999)
    2. MI threshold gate for partnership qualification
    3. Behavioral correlation detection for Sybil resistance
    4. View change protocol for primary recovery

    BFT Invariant: n >= 3f + 1 where:
    - n = total nodes
    - f = maximum Byzantine (malicious) nodes

    Usage:
        params = FederationParams(num_honest=7, num_malicious=2)
        engine = FederationEngine(params)

        # Initialize federation
        await engine.initialize()

        # Submit precedent for consensus
        result = await engine.propose_precedent(precedent)

        # Get federation state
        state = engine.get_federation_state()
    """

    def __init__(
        self,
        params: FederationParams,
        config: Optional[BFTConfig] = None,
    ):
        """
        Initialize federation engine.

        Args:
            params: Federation parameters (from schemas.simulation)
            config: PBFT configuration (optional, uses defaults)

        Raises:
            ValueError: If BFT invariant is violated
        """
        self.params = params
        self.config = config or BFTConfig()

        # Validate BFT invariant
        self.n = params.num_honest + params.num_malicious
        self.f = params.num_malicious

        if not verify_bft_invariant(self.n, self.f):
            raise ValueError(
                f"BFT invariant violated: n={self.n} must be >= 3f+1={3*self.f+1} "
                f"to tolerate f={self.f} Byzantine failures"
            )

        # Compute quorum size
        self.quorum_size = compute_quorum_size(self.n)

        # Node management
        self.nodes: Dict[str, FederationNode] = {}
        self.honest_nodes: List[str] = []
        self.malicious_nodes: List[str] = []

        # Partnership
        self.partners: Set[str] = set()
        self.mi_gate = MIThresholdGate(mi_threshold=params.mi_threshold)

        # Sybil detection
        self.sybil_detector = BehavioralCorrelationDetector()
        self.correlation_detection_enabled = params.correlation_detection_enabled

        # Consensus state
        self.current_view = 0
        self.current_sequence = 0
        self.last_executed_sequence = 0

        # Active consensus rounds
        self.rounds: Dict[int, ConsensusRound] = {}

        # Primary rotation
        self.primary_id: Optional[str] = None

        # Precedent storage
        self.precedents: Dict[str, Precedent] = {}

        # Metrics
        self.metrics = FederationMetrics()

        # Running state
        self._running = False

    async def initialize(self) -> None:
        """
        Initialize the federation with nodes.

        Creates honest and malicious nodes with:
        - Unique identifiers
        - Cryptographic keypairs
        - Network transports
        - Storage backends
        """
        # Reset transport registry
        InMemoryTransport.reset_transports()

        # Create honest nodes
        for i in range(self.params.num_honest):
            node_id = f"honest-{i}"
            node = await self._create_node(node_id, NodeType.HONEST)
            self.nodes[node_id] = node
            self.honest_nodes.append(node_id)

        # Create malicious nodes
        for i in range(self.params.num_malicious):
            node_id = f"malicious-{i}"
            node = await self._create_node(
                node_id,
                NodeType.MALICIOUS,
                malicious_strategy=self.params.malicious_strategy
            )
            self.nodes[node_id] = node
            self.malicious_nodes.append(node_id)

        # Set initial primary (round-robin based on view)
        all_node_ids = list(self.nodes.keys())
        self.primary_id = all_node_ids[self.current_view % len(all_node_ids)]

        # Start all transports
        for node in self.nodes.values():
            await node.transport.start()

        self._running = True

    async def _create_node(
        self,
        node_id: str,
        node_type: NodeType,
        malicious_strategy: Optional[MaliciousStrategy] = None,
    ) -> FederationNode:
        """Create a federation node with all backends."""
        crypto = Ed25519CryptoProvider.generate(node_id)
        transport = InMemoryTransport(node_id)
        storage = InMemoryStorage()

        return FederationNode(
            node_id=node_id,
            node_type=node_type,
            crypto=crypto,
            transport=transport,
            storage=storage,
            malicious_strategy=malicious_strategy,
        )

    async def shutdown(self) -> None:
        """Shutdown the federation."""
        self._running = False
        for node in self.nodes.values():
            await node.transport.stop()

    # =========================================================================
    # PBFT Consensus
    # =========================================================================

    async def propose_precedent(
        self,
        content: str,
        proposer_id: str,
    ) -> Optional[Precedent]:
        """
        Propose a new precedent for consensus.

        Args:
            content: The precedent content
            proposer_id: ID of the proposing node

        Returns:
            The accepted Precedent if consensus achieved, None otherwise
        """
        if proposer_id not in self.nodes:
            raise ValueError(f"Unknown proposer: {proposer_id}")

        # Create precedent
        precedent_id = str(uuid4())
        precedent = Precedent(
            id=precedent_id,
            content=content,
            proposer_id=proposer_id,
            votes=[],
            status="proposed",
            timestamp=time.time(),
        )

        # Create request
        request = Request(
            operation={
                "type": "ADD_PRECEDENT",
                "precedent_id": precedent_id,
                "content": content,
            },
            client_id=proposer_id,
        )

        # Run consensus
        success = await self._run_consensus(request)

        if success:
            precedent.status = "approved"
            self.precedents[precedent_id] = precedent
            return precedent
        else:
            precedent.status = "rejected"
            return None

    async def _run_consensus(self, request: Request) -> bool:
        """
        Run PBFT consensus for a request.

        Implements the full PBFT protocol:
        1. PRE-PREPARE (primary)
        2. PREPARE (all replicas)
        3. COMMIT (all replicas)
        4. REPLY (execution)

        Args:
            request: The client request

        Returns:
            True if consensus achieved, False otherwise
        """
        self.current_sequence += 1
        sequence = self.current_sequence

        # Compute request digest
        request_digest = Digest.from_message(request)

        # Initialize round
        round_state = ConsensusRound(
            sequence_number=sequence,
            view_number=self.current_view,
            request=request,
            request_digest=request_digest,
        )
        self.rounds[sequence] = round_state

        # Phase 1: PRE-PREPARE (primary sends to all)
        pre_prepare = await self._send_pre_prepare(round_state)
        if pre_prepare is None:
            return False
        round_state.pre_prepare = pre_prepare

        # Phase 2: PREPARE (all replicas)
        prepare_success = await self._collect_prepares(round_state)
        if not prepare_success:
            return False
        round_state.prepared = True

        # Phase 3: COMMIT (all replicas)
        commit_success = await self._collect_commits(round_state)
        if not commit_success:
            return False
        round_state.committed_local = True

        # Phase 4: Execute
        await self._execute(round_state)
        round_state.executed = True
        round_state.completed_at = time.time()

        self.last_executed_sequence = sequence
        self.metrics.successful_rounds += 1
        self.metrics.total_rounds += 1

        return True

    async def _send_pre_prepare(
        self,
        round_state: ConsensusRound,
    ) -> Optional[PrePrepare]:
        """Primary sends PRE-PREPARE to all replicas."""
        if self.primary_id is None:
            return None

        primary = self.nodes[self.primary_id]

        # Create PRE-PREPARE message
        pre_prepare = PrePrepare(
            view_number=round_state.view_number,
            sequence_number=round_state.sequence_number,
            request_digest=round_state.request_digest,
            request=round_state.request,
            primary_id=self.primary_id,
        )

        # Sign it
        message_bytes = pre_prepare.model_dump_json(exclude={'signature'}).encode()
        signature = primary.crypto.sign(message_bytes)
        pre_prepare.signature = signature

        # Broadcast to all replicas
        recipients = [nid for nid in self.nodes.keys() if nid != self.primary_id]
        await primary.transport.broadcast(pre_prepare, recipients)

        return pre_prepare

    async def _collect_prepares(
        self,
        round_state: ConsensusRound,
    ) -> bool:
        """Collect PREPARE messages from replicas."""
        # Each replica (except primary) sends PREPARE
        for node_id, node in self.nodes.items():
            if node_id == self.primary_id:
                continue

            # Malicious nodes may not send PREPARE
            if node.node_type == NodeType.MALICIOUS:
                if not self._should_malicious_participate(node):
                    continue

            # Create PREPARE
            prepare = Prepare(
                view_number=round_state.view_number,
                sequence_number=round_state.sequence_number,
                request_digest=round_state.request_digest,
                replica_id=node_id,
            )

            # Sign it
            message_bytes = prepare.model_dump_json(exclude={'signature'}).encode()
            signature = node.crypto.sign(message_bytes)
            prepare.signature = signature

            round_state.prepares[node_id] = prepare

            # Update behavioral trace
            node.behavioral_trace.append(1.0)  # Participated

        # Check if we have 2f + 1 prepares (including primary)
        # The primary counts as one prepare implicitly
        prepare_count = len(round_state.prepares) + 1  # +1 for primary

        return prepare_count >= self.quorum_size

    async def _collect_commits(
        self,
        round_state: ConsensusRound,
    ) -> bool:
        """Collect COMMIT messages from replicas."""
        # Each replica sends COMMIT after prepared
        for node_id, node in self.nodes.items():
            # Malicious nodes may not send COMMIT
            if node.node_type == NodeType.MALICIOUS:
                if not self._should_malicious_participate(node):
                    continue

            # Create COMMIT
            commit = Commit(
                view_number=round_state.view_number,
                sequence_number=round_state.sequence_number,
                request_digest=round_state.request_digest,
                replica_id=node_id,
            )

            # Sign it
            message_bytes = commit.model_dump_json(exclude={'signature'}).encode()
            signature = node.crypto.sign(message_bytes)
            commit.signature = signature

            round_state.commits[node_id] = commit

            # Update behavioral trace
            node.behavioral_trace.append(1.0)  # Participated

        # Check if we have 2f + 1 commits
        return len(round_state.commits) >= self.quorum_size

    async def _execute(self, round_state: ConsensusRound) -> None:
        """Execute the request after consensus."""
        # Store in all honest nodes
        for node_id in self.honest_nodes:
            node = self.nodes[node_id]
            log = MessageLog(
                sequence_number=round_state.sequence_number,
                view_number=round_state.view_number,
                request=round_state.request,
                pre_prepare=round_state.pre_prepare,
                prepares=round_state.prepares,
                commits=round_state.commits,
                prepared=round_state.prepared,
                committed_local=round_state.committed_local,
                executed=True,
            )
            await node.storage.store_message_log(
                round_state.view_number,
                round_state.sequence_number,
                log,
            )

    def _should_malicious_participate(self, node: FederationNode) -> bool:
        """Determine if a malicious node participates (based on strategy)."""
        strategy = node.malicious_strategy or self.params.malicious_strategy

        if strategy == MaliciousStrategy.RANDOM:
            # Random participation
            return random.random() > 0.3

        elif strategy == MaliciousStrategy.COORDINATED:
            # Coordinated non-participation to try to block consensus
            return False

        elif strategy == MaliciousStrategy.SLOW_CAPTURE:
            # Initially participate, then stop
            return len(node.behavioral_trace) < 10

        return True

    # =========================================================================
    # Partnership Management
    # =========================================================================

    async def apply_for_partnership(
        self,
        node_id: str,
    ) -> Tuple[bool, str]:
        """
        Process a partnership application.

        Args:
            node_id: ID of node applying for partnership

        Returns:
            Tuple of (accepted, reason)
        """
        if node_id not in self.nodes:
            return (False, "Unknown node")

        node = self.nodes[node_id]

        # Check MI threshold
        partner_nodes = [self.nodes[pid] for pid in self.partners]
        qualifies, avg_mi = self.mi_gate.qualifies_for_partnership(node, partner_nodes)

        if not qualifies:
            return (False, f"MI threshold exceeded: avg_mi={avg_mi:.3f}")

        # Check for Sybil correlation (if enabled)
        if self.correlation_detection_enabled:
            all_nodes = list(self.nodes.values())
            is_suspicious, max_corr = self.sybil_detector.is_suspicious(node, all_nodes)

            if is_suspicious:
                return (False, f"Suspicious correlation detected: {max_corr:.3f}")

        # Accept partnership
        self.partners.add(node_id)
        node.is_partner = True
        node.mi_value = avg_mi

        return (True, "Partnership granted")

    def detect_sybil_clusters(self) -> List[Set[str]]:
        """Run Sybil detection across all nodes."""
        all_nodes = list(self.nodes.values())
        return self.sybil_detector.detect_sybil_clusters(all_nodes)

    # =========================================================================
    # View Change
    # =========================================================================

    async def initiate_view_change(self, initiator_id: str) -> bool:
        """
        Initiate a view change (leader election).

        Called when primary is suspected faulty.

        Args:
            initiator_id: ID of node initiating view change

        Returns:
            True if view change succeeded
        """
        new_view = self.current_view + 1

        # Collect VIEW-CHANGE messages
        view_change_messages: Dict[str, ViewChange] = {}

        for node_id, node in self.nodes.items():
            if node.node_type == NodeType.MALICIOUS:
                if not self._should_malicious_participate(node):
                    continue

            vc = ViewChange(
                new_view_number=new_view,
                last_stable_checkpoint=0,  # Simplified
                replica_id=node_id,
            )
            view_change_messages[node_id] = vc

        # Need 2f + 1 view changes
        if len(view_change_messages) < self.quorum_size:
            return False

        # New primary (round-robin)
        all_node_ids = list(self.nodes.keys())
        new_primary_id = all_node_ids[new_view % len(all_node_ids)]

        # Update state
        self.current_view = new_view
        self.primary_id = new_primary_id

        self.metrics.view_changes += 1

        return True

    # =========================================================================
    # State and Metrics
    # =========================================================================

    def get_federation_state(self) -> FederationState:
        """Get current federation state."""
        return FederationState(
            view_number=self.current_view,
            last_sequence=self.last_executed_sequence,
            total_replicas=self.n,
            max_faulty=self.f,
            replica_ids=list(self.nodes.keys()),
            primary_id=self.primary_id or "",
            accepted_precedents=[UUID(pid) for pid in self.precedents.keys()],
            pending_precedents=[],
        )

    def get_metrics(self) -> 'FederationMetrics':
        """Get federation metrics."""
        return self.metrics

    def validate_bft_invariant(self) -> bool:
        """
        Validate the BFT invariant: n >= 3f + 1

        Returns:
            True if invariant holds
        """
        return verify_bft_invariant(self.n, self.f)


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class FederationMetrics:
    """Metrics for federation health and performance."""
    total_rounds: int = 0
    successful_rounds: int = 0
    failed_rounds: int = 0
    view_changes: int = 0
    sybil_clusters_detected: int = 0
    partnership_rejections: int = 0

    @property
    def success_rate(self) -> float:
        """Consensus success rate."""
        if self.total_rounds == 0:
            return 0.0
        return self.successful_rounds / self.total_rounds


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_federation(
    params: FederationParams,
    config: Optional[BFTConfig] = None,
) -> FederationEngine:
    """
    Factory function to create and initialize a federation.

    Args:
        params: Federation parameters
        config: Optional PBFT configuration

    Returns:
        Initialized FederationEngine
    """
    engine = FederationEngine(params, config)
    await engine.initialize()
    return engine


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'FederationEngine',
    'FederationNode',
    'FederationMetrics',
    'NodeType',
    'ConsensusRound',
    'BehavioralCorrelationDetector',
    'MIThresholdGate',
    'create_federation',
]
