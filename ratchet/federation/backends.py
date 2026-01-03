"""
RATCHET BFT Backend Implementations

Concrete implementations of the abstract interfaces defined in schemas/bft.py:
- InMemoryTransport: Asyncio queue-based message passing for testing
- InMemoryStorage: Dict-based storage for logs and checkpoints
- Ed25519CryptoProvider: Ed25519 signing/verification using cryptography library

These backends enable testing and simulation of PBFT consensus without
requiring actual network or persistent storage infrastructure.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from schemas.bft import (
    CheckpointMessage,
    CryptoProvider,
    Digest,
    MessageLog,
    MessageType,
    NetworkTransport,
    PBFTMessage,
    PersistentStorage,
    Signature,
)


# =============================================================================
# IN-MEMORY TRANSPORT
# =============================================================================

class InMemoryTransport(NetworkTransport):
    """
    In-memory transport using asyncio queues for message passing.

    Designed for testing and simulation of PBFT consensus without
    actual network infrastructure. Messages are delivered reliably
    and in FIFO order per sender.

    Features:
    - Asyncio queues for non-blocking message delivery
    - Optional message delay simulation for latency testing
    - Message handler registration for event-driven processing
    - Support for network partitioning simulation

    Usage:
        transport = InMemoryTransport(replica_id="node-0")
        await transport.start()

        # Register handler
        transport.register_handler(MessageType.PREPARE, handle_prepare)

        # Send messages
        await transport.send(message, "node-1")

        # Or receive directly
        sender, msg = await transport.receive(timeout_ms=1000)
    """

    # Class-level registry of all transports for routing
    _transports: Dict[str, 'InMemoryTransport'] = {}

    def __init__(
        self,
        replica_id: str,
        simulated_latency_ms: float = 0.0,
    ):
        """
        Initialize in-memory transport.

        Args:
            replica_id: Unique identifier for this replica
            simulated_latency_ms: Optional delay for each message (for testing)
        """
        self._replica_id = replica_id
        self._simulated_latency_ms = simulated_latency_ms
        self._inbox: asyncio.Queue[Tuple[str, PBFTMessage]] = asyncio.Queue()
        self._handlers: Dict[MessageType, List[Callable[[str, PBFTMessage], Awaitable[None]]]] = defaultdict(list)
        self._running = False
        self._handler_task: Optional[asyncio.Task] = None
        self._partitioned_from: Set[str] = set()  # Simulated network partitions

    @classmethod
    def reset_transports(cls) -> None:
        """Reset all transports (for testing)."""
        cls._transports.clear()

    @classmethod
    def get_transport(cls, replica_id: str) -> Optional['InMemoryTransport']:
        """Get transport by replica ID."""
        return cls._transports.get(replica_id)

    async def send(
        self,
        message: PBFTMessage,
        recipient: str,
    ) -> bool:
        """
        Send a message to a specific replica.

        Args:
            message: The PBFT message to send
            recipient: The replica ID of the recipient

        Returns:
            True if message was queued successfully, False if recipient unknown
            or partitioned
        """
        # Check for simulated partition
        if recipient in self._partitioned_from:
            return False

        target = self._transports.get(recipient)
        if target is None:
            return False

        # Simulate network latency
        if self._simulated_latency_ms > 0:
            await asyncio.sleep(self._simulated_latency_ms / 1000.0)

        # Deliver to recipient's inbox
        await target._inbox.put((self._replica_id, message))
        return True

    async def broadcast(
        self,
        message: PBFTMessage,
        recipients: List[str],
    ) -> Dict[str, bool]:
        """
        Broadcast a message to multiple replicas.

        Args:
            message: The PBFT message to broadcast
            recipients: List of replica IDs to send to

        Returns:
            Dictionary mapping recipient ID to send success status
        """
        results = {}
        # Send in parallel
        tasks = [self.send(message, recipient) for recipient in recipients]
        send_results = await asyncio.gather(*tasks)
        for recipient, success in zip(recipients, send_results):
            results[recipient] = success
        return results

    async def receive(
        self,
        timeout_ms: Optional[int] = None,
    ) -> Optional[Tuple[str, PBFTMessage]]:
        """
        Receive the next message from any sender.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.
                        None means wait indefinitely.

        Returns:
            Tuple of (sender_id, message) if message received,
            None if timeout occurred.
        """
        try:
            if timeout_ms is None:
                result = await self._inbox.get()
            else:
                result = await asyncio.wait_for(
                    self._inbox.get(),
                    timeout=timeout_ms / 1000.0
                )
            return result
        except asyncio.TimeoutError:
            return None

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[str, PBFTMessage], Awaitable[None]],
    ) -> None:
        """
        Register a callback handler for a specific message type.

        Args:
            message_type: The type of message to handle
            handler: Async callback function taking (sender_id, message)
        """
        self._handlers[message_type].append(handler)

    async def _message_handler_loop(self) -> None:
        """Internal loop that dispatches messages to registered handlers."""
        while self._running:
            try:
                result = await asyncio.wait_for(self._inbox.get(), timeout=0.1)
                sender_id, message = result

                # Dispatch to handlers
                msg_type = message.message_type
                for handler in self._handlers.get(msg_type, []):
                    try:
                        await handler(sender_id, message)
                    except Exception as e:
                        # Log but don't crash the handler loop
                        print(f"Handler error for {msg_type}: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def start(self) -> None:
        """
        Start the transport layer.

        Registers this transport and optionally starts the handler loop.
        """
        self._transports[self._replica_id] = self
        self._running = True

        # Start handler loop if handlers are registered
        if self._handlers:
            self._handler_task = asyncio.create_task(self._message_handler_loop())

    async def stop(self) -> None:
        """
        Stop the transport layer.

        Gracefully shuts down the handler loop and unregisters.
        """
        self._running = False

        if self._handler_task is not None:
            self._handler_task.cancel()
            try:
                await self._handler_task
            except asyncio.CancelledError:
                pass
            self._handler_task = None

        # Unregister
        if self._replica_id in self._transports:
            del self._transports[self._replica_id]

    def get_replica_id(self) -> str:
        """Return the ID of this replica."""
        return self._replica_id

    # ==========================================================================
    # Simulation Helpers
    # ==========================================================================

    def simulate_partition(self, other_replica: str) -> None:
        """Simulate network partition from this replica to another."""
        self._partitioned_from.add(other_replica)

    def heal_partition(self, other_replica: str) -> None:
        """Heal a simulated network partition."""
        self._partitioned_from.discard(other_replica)

    def set_latency(self, latency_ms: float) -> None:
        """Set simulated latency for outgoing messages."""
        self._simulated_latency_ms = latency_ms


# =============================================================================
# IN-MEMORY STORAGE
# =============================================================================

class InMemoryStorage(PersistentStorage):
    """
    In-memory storage for PBFT message logs and checkpoints.

    Uses Python dicts for storage. Suitable for testing and simulation
    but NOT for production (no durability guarantees).

    Storage structure:
    - Message logs: Dict[(view, sequence)] -> MessageLog
    - Checkpoints: Dict[sequence] -> (CheckpointMessage, state_bytes)
    - View state: (view_number, primary_id, last_executed_sequence)

    Thread safety: Not thread-safe. Use with asyncio only.

    Usage:
        storage = InMemoryStorage()

        # Store and retrieve logs
        await storage.store_message_log(view=0, sequence=1, log=log)
        log = await storage.get_message_log(view=0, sequence=1)

        # Store checkpoints
        await storage.store_checkpoint(
            sequence_number=100,
            checkpoint=checkpoint_msg,
            state_snapshot=state_bytes
        )
    """

    def __init__(self):
        """Initialize empty storage."""
        # Message logs indexed by (view_number, sequence_number)
        self._message_logs: Dict[Tuple[int, int], MessageLog] = {}

        # Checkpoints indexed by sequence_number
        self._checkpoints: Dict[int, Tuple[CheckpointMessage, bytes]] = {}

        # Current view state
        self._view_state: Optional[Tuple[int, str, int]] = None

        # Stable checkpoint sequence number
        self._stable_checkpoint_seq: Optional[int] = None

    async def store_message_log(
        self,
        view_number: int,
        sequence_number: int,
        log: MessageLog,
    ) -> None:
        """
        Store a message log entry.

        Args:
            view_number: The view number for this log entry
            sequence_number: The sequence number for this log entry
            log: The message log to store
        """
        self._message_logs[(view_number, sequence_number)] = log

    async def get_message_log(
        self,
        view_number: int,
        sequence_number: int,
    ) -> Optional[MessageLog]:
        """
        Retrieve a message log entry.

        Args:
            view_number: The view number to look up
            sequence_number: The sequence number to look up

        Returns:
            The stored MessageLog, or None if not found
        """
        return self._message_logs.get((view_number, sequence_number))

    async def get_message_logs_range(
        self,
        view_number: int,
        start_sequence: int,
        end_sequence: int,
    ) -> List[MessageLog]:
        """
        Retrieve message logs for a range of sequence numbers.

        Args:
            view_number: The view number to query
            start_sequence: Start of sequence range (inclusive)
            end_sequence: End of sequence range (inclusive)

        Returns:
            List of MessageLogs in the specified range, ordered by sequence
        """
        logs = []
        for seq in range(start_sequence, end_sequence + 1):
            log = self._message_logs.get((view_number, seq))
            if log is not None:
                logs.append(log)
        return logs

    async def store_checkpoint(
        self,
        sequence_number: int,
        checkpoint: CheckpointMessage,
        state_snapshot: bytes,
    ) -> None:
        """
        Store a checkpoint with its associated state snapshot.

        Args:
            sequence_number: The sequence number of the checkpoint
            checkpoint: The checkpoint message
            state_snapshot: Serialized application state at this checkpoint
        """
        self._checkpoints[sequence_number] = (checkpoint, state_snapshot)

    async def get_checkpoint(
        self,
        sequence_number: int,
    ) -> Optional[Tuple[CheckpointMessage, bytes]]:
        """
        Retrieve a checkpoint and its state snapshot.

        Args:
            sequence_number: The sequence number to look up

        Returns:
            Tuple of (CheckpointMessage, state_snapshot) or None if not found
        """
        return self._checkpoints.get(sequence_number)

    async def get_stable_checkpoint(self) -> Optional[Tuple[int, CheckpointMessage, bytes]]:
        """
        Get the most recent stable checkpoint.

        Returns:
            Tuple of (sequence_number, CheckpointMessage, state_snapshot)
            or None if no stable checkpoint exists
        """
        if self._stable_checkpoint_seq is None:
            return None

        checkpoint_data = self._checkpoints.get(self._stable_checkpoint_seq)
        if checkpoint_data is None:
            return None

        checkpoint, state = checkpoint_data
        return (self._stable_checkpoint_seq, checkpoint, state)

    def mark_checkpoint_stable(self, sequence_number: int) -> None:
        """Mark a checkpoint as stable (received 2f+1 matching checkpoints)."""
        self._stable_checkpoint_seq = sequence_number

    async def store_view_state(
        self,
        view_number: int,
        primary_id: str,
        last_executed_sequence: int,
    ) -> None:
        """
        Store the current view state.

        Args:
            view_number: Current view number
            primary_id: ID of the current primary
            last_executed_sequence: Last executed sequence number
        """
        self._view_state = (view_number, primary_id, last_executed_sequence)

    async def get_view_state(self) -> Optional[Tuple[int, str, int]]:
        """
        Retrieve the stored view state.

        Returns:
            Tuple of (view_number, primary_id, last_executed_sequence)
            or None if no state stored (fresh start)
        """
        return self._view_state

    async def garbage_collect(
        self,
        stable_checkpoint_sequence: int,
    ) -> int:
        """
        Remove message logs older than the stable checkpoint.

        Args:
            stable_checkpoint_sequence: Sequence number of stable checkpoint.
                                        All logs with sequence < this are removed.

        Returns:
            Number of entries removed
        """
        removed_count = 0
        keys_to_remove = []

        for (view, seq) in self._message_logs.keys():
            if seq < stable_checkpoint_sequence:
                keys_to_remove.append((view, seq))

        for key in keys_to_remove:
            del self._message_logs[key]
            removed_count += 1

        # Also clean up old checkpoints
        checkpoint_keys_to_remove = [
            seq for seq in self._checkpoints.keys()
            if seq < stable_checkpoint_sequence
        ]
        for key in checkpoint_keys_to_remove:
            del self._checkpoints[key]
            removed_count += 1

        return removed_count

    async def clear(self) -> None:
        """
        Clear all stored data.

        WARNING: This is destructive and should only be used for testing
        or when intentionally resetting a replica.
        """
        self._message_logs.clear()
        self._checkpoints.clear()
        self._view_state = None
        self._stable_checkpoint_seq = None


# =============================================================================
# ED25519 CRYPTO PROVIDER
# =============================================================================

class Ed25519CryptoProvider(CryptoProvider):
    """
    Cryptographic provider using Ed25519 signatures.

    Uses the cryptography library for Ed25519 key generation, signing,
    and verification. Provides 128-bit security level.

    Features:
    - Ed25519 digital signatures (RFC 8032)
    - SHA-256 message digests
    - Deterministic signatures (same message always produces same signature)
    - Fast verification (important for PBFT with many messages)

    Usage:
        # Generate new provider
        crypto = Ed25519CryptoProvider.generate(replica_id="node-0")

        # Or from existing key
        crypto = Ed25519CryptoProvider(private_key, replica_id="node-0")

        # Sign a message
        signature = crypto.sign(message_bytes)

        # Verify a signature
        is_valid = crypto.verify(signature, message_bytes, signer_public_key)
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey,
        replica_id: str,
    ):
        """
        Initialize crypto provider with an existing private key.

        Args:
            private_key: Ed25519 private key
            replica_id: ID of this replica (for signature metadata)
        """
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._replica_id = replica_id

    @classmethod
    def generate(cls, replica_id: str) -> 'Ed25519CryptoProvider':
        """
        Generate a new crypto provider with a fresh keypair.

        Args:
            replica_id: ID of this replica

        Returns:
            New Ed25519CryptoProvider instance
        """
        private_key = Ed25519PrivateKey.generate()
        return cls(private_key, replica_id)

    @classmethod
    def from_private_key_bytes(
        cls,
        private_key_bytes: bytes,
        replica_id: str,
    ) -> 'Ed25519CryptoProvider':
        """
        Create crypto provider from private key bytes.

        Args:
            private_key_bytes: Raw Ed25519 private key (32 bytes)
            replica_id: ID of this replica

        Returns:
            Ed25519CryptoProvider instance
        """
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        return cls(private_key, replica_id)

    def sign(
        self,
        message_bytes: bytes,
    ) -> Signature:
        """
        Sign a message with this replica's private key.

        Args:
            message_bytes: The bytes to sign

        Returns:
            A Signature object containing the signature value
        """
        signature_bytes = self._private_key.sign(message_bytes)
        signature_b64 = base64.b64encode(signature_bytes).decode('ascii')

        return Signature(
            signer_id=self._replica_id,
            algorithm="ed25519",
            value=signature_b64,
            timestamp_ms=int(time.time() * 1000),
        )

    def verify(
        self,
        signature: Signature,
        message_bytes: bytes,
        signer_public_key: bytes,
    ) -> bool:
        """
        Verify a signature against a message.

        Args:
            signature: The signature to verify
            message_bytes: The original message bytes
            signer_public_key: The public key of the claimed signer (32 bytes)

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Decode signature from base64
            signature_bytes = base64.b64decode(signature.value)

            # Load the public key
            public_key = Ed25519PublicKey.from_public_bytes(signer_public_key)

            # Verify - raises InvalidSignature if invalid
            public_key.verify(signature_bytes, message_bytes)
            return True

        except (InvalidSignature, ValueError, Exception):
            return False

    def get_public_key(self) -> bytes:
        """
        Get this replica's public key.

        Returns:
            The public key as bytes (32 bytes for Ed25519)
        """
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def compute_digest(
        self,
        data: bytes,
    ) -> Digest:
        """
        Compute a cryptographic digest of data.

        Args:
            data: The bytes to hash

        Returns:
            A Digest object containing the SHA-256 hash value
        """
        hash_value = hashlib.sha256(data).hexdigest()
        return Digest(algorithm="sha256", value=hash_value)

    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generate a new keypair.

        Returns:
            Tuple of (private_key_bytes, public_key_bytes)
        """
        new_private_key = Ed25519PrivateKey.generate()

        private_bytes = new_private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_bytes = new_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        return (private_bytes, public_bytes)

    def get_replica_id(self) -> str:
        """Return the replica ID associated with this crypto provider."""
        return self._replica_id


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'InMemoryTransport',
    'InMemoryStorage',
    'Ed25519CryptoProvider',
]
