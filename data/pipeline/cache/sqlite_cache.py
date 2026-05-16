"""
SQLite Cache for RATCHET Data Pipeline

Provides persistent caching of fetched data with:
    - Time-to-live (TTL) based expiration
    - Manual invalidation support
    - Query by source, series, and date range
    - Automatic cleanup of expired entries
"""

from __future__ import annotations

import json
import sqlite3
import hashlib
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import io

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    key: str
    source: str
    series_id: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    created_at: datetime
    expires_at: datetime
    size_bytes: int
    row_count: int
    metadata: Dict[str, Any]

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()


class SQLiteCache:
    """
    SQLite-based cache for fetched data.

    Features:
        - Persistent storage across sessions
        - TTL-based automatic expiration
        - Date range queries for partial cache hits
        - Automatic cleanup of expired entries
        - Thread-safe operations

    Example:
        >>> cache = SQLiteCache("/path/to/cache.db", default_ttl_hours=24)
        >>> cache.put("fred", "GDP", df, start_date, end_date)
        >>> cached_df = cache.get("fred", "GDP", start_date, end_date)
        >>> print(f"Cache size: {cache.size_mb:.1f} MB")
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: Union[str, Path] = "~/.ratchet/data_cache.db",
        default_ttl_hours: int = 24,
        max_size_mb: Optional[float] = 1000,
        auto_cleanup: bool = True,
    ):
        """
        Initialize the cache.

        Args:
            db_path: Path to SQLite database file
            default_ttl_hours: Default time-to-live for cache entries
            max_size_mb: Maximum cache size in MB (None for unlimited)
            auto_cleanup: Whether to run cleanup on startup
        """
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_size_mb = max_size_mb

        self._init_db()

        if auto_cleanup:
            self.cleanup_expired()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create main cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    start_date TEXT,
                    end_date TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    data BLOB NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    row_count INTEGER NOT NULL,
                    metadata TEXT
                )
            """)

            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_series
                ON cache_entries (source, series_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at
                ON cache_entries (expires_at)
            """)

            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            # Store schema version
            cursor.execute("""
                INSERT OR REPLACE INTO cache_metadata (key, value)
                VALUES ('schema_version', ?)
            """, (str(self.SCHEMA_VERSION),))

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _make_key(
        self,
        source: str,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Generate a cache key from parameters."""
        parts = [source, series_id]
        if start_date:
            parts.append(start_date.strftime('%Y-%m-%d'))
        if end_date:
            parts.append(end_date.strftime('%Y-%m-%d'))

        key_string = "|".join(parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _serialize_dataframe(self, df: pd.DataFrame) -> bytes:
        """Serialize DataFrame to bytes."""
        buffer = io.BytesIO()
        df.to_parquet(buffer, engine='pyarrow', compression='snappy')
        return buffer.getvalue()

    def _deserialize_dataframe(self, data: bytes) -> pd.DataFrame:
        """Deserialize bytes to DataFrame."""
        buffer = io.BytesIO(data)
        return pd.read_parquet(buffer)

    def put(
        self,
        source: str,
        series_id: str,
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store data in the cache.

        Args:
            source: Data source identifier
            series_id: Series identifier
            data: DataFrame to cache
            start_date: Start date of data range
            end_date: End date of data range
            ttl: Time-to-live (defaults to default_ttl_hours)
            metadata: Additional metadata to store

        Returns:
            Cache key for the entry
        """
        key = self._make_key(source, series_id, start_date, end_date)
        ttl = ttl or self.default_ttl

        now = datetime.utcnow()
        expires_at = now + ttl

        serialized = self._serialize_dataframe(data)
        size_bytes = len(serialized)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO cache_entries
                (key, source, series_id, start_date, end_date, created_at,
                 expires_at, data, size_bytes, row_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                source,
                series_id,
                start_date.isoformat() if start_date else None,
                end_date.isoformat() if end_date else None,
                now.isoformat(),
                expires_at.isoformat(),
                serialized,
                size_bytes,
                len(data),
                json.dumps(metadata or {}),
            ))

            conn.commit()

        logger.debug(f"Cached {source}/{series_id}: {len(data)} rows, {size_bytes} bytes")

        # Check if cleanup is needed
        if self.max_size_mb and self.size_mb > self.max_size_mb:
            self._evict_oldest()

        return key

    def get(
        self,
        source: str,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ignore_expiry: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve data from the cache.

        Args:
            source: Data source identifier
            series_id: Series identifier
            start_date: Start date of data range
            end_date: End date of data range
            ignore_expiry: If True, return expired entries too

        Returns:
            Cached DataFrame or None if not found/expired
        """
        key = self._make_key(source, series_id, start_date, end_date)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT data, expires_at FROM cache_entries
                WHERE key = ?
            """, (key,))

            row = cursor.fetchone()

            if row is None:
                return None

            expires_at = datetime.fromisoformat(row['expires_at'])
            if not ignore_expiry and datetime.utcnow() > expires_at:
                # Entry expired, delete it
                cursor.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                return None

            return self._deserialize_dataframe(row['data'])

    def get_entry(
        self,
        source: str,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[CacheEntry]:
        """
        Get cache entry metadata without loading data.

        Args:
            source: Data source identifier
            series_id: Series identifier
            start_date: Start date
            end_date: End date

        Returns:
            CacheEntry or None if not found
        """
        key = self._make_key(source, series_id, start_date, end_date)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT key, source, series_id, start_date, end_date,
                       created_at, expires_at, size_bytes, row_count, metadata
                FROM cache_entries
                WHERE key = ?
            """, (key,))

            row = cursor.fetchone()

            if row is None:
                return None

            return CacheEntry(
                key=row['key'],
                source=row['source'],
                series_id=row['series_id'],
                start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else None,
                end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                created_at=datetime.fromisoformat(row['created_at']),
                expires_at=datetime.fromisoformat(row['expires_at']),
                size_bytes=row['size_bytes'],
                row_count=row['row_count'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
            )

    def invalidate(
        self,
        source: Optional[str] = None,
        series_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Args:
            source: Invalidate all entries from this source
            series_id: Invalidate entries for this series (requires source)

        Returns:
            Number of entries invalidated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if source and series_id:
                cursor.execute("""
                    DELETE FROM cache_entries
                    WHERE source = ? AND series_id = ?
                """, (source, series_id))
            elif source:
                cursor.execute("""
                    DELETE FROM cache_entries
                    WHERE source = ?
                """, (source,))
            else:
                cursor.execute("DELETE FROM cache_entries")

            deleted = cursor.rowcount
            conn.commit()

        logger.info(f"Invalidated {deleted} cache entries")
        return deleted

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.utcnow().isoformat()
            cursor.execute("""
                DELETE FROM cache_entries
                WHERE expires_at < ?
            """, (now,))

            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired cache entries")

        return deleted

    def _evict_oldest(self) -> int:
        """
        Evict oldest entries to stay under size limit.

        Returns:
            Number of entries evicted
        """
        if not self.max_size_mb:
            return 0

        evicted = 0
        target_bytes = int(self.max_size_mb * 0.8 * 1024 * 1024)  # Target 80% of max

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get current size
            cursor.execute("SELECT SUM(size_bytes) as total FROM cache_entries")
            current_size = cursor.fetchone()['total'] or 0

            while current_size > target_bytes:
                # Delete oldest entry
                cursor.execute("""
                    DELETE FROM cache_entries
                    WHERE key = (
                        SELECT key FROM cache_entries
                        ORDER BY created_at ASC
                        LIMIT 1
                    )
                """)

                if cursor.rowcount == 0:
                    break

                evicted += cursor.rowcount

                # Recalculate size
                cursor.execute("SELECT SUM(size_bytes) as total FROM cache_entries")
                current_size = cursor.fetchone()['total'] or 0

            conn.commit()

        if evicted > 0:
            logger.info(f"Evicted {evicted} entries to stay under size limit")

        return evicted

    def list_entries(
        self,
        source: Optional[str] = None,
    ) -> List[CacheEntry]:
        """
        List all cache entries.

        Args:
            source: Filter by source

        Returns:
            List of CacheEntry objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if source:
                cursor.execute("""
                    SELECT key, source, series_id, start_date, end_date,
                           created_at, expires_at, size_bytes, row_count, metadata
                    FROM cache_entries
                    WHERE source = ?
                    ORDER BY created_at DESC
                """, (source,))
            else:
                cursor.execute("""
                    SELECT key, source, series_id, start_date, end_date,
                           created_at, expires_at, size_bytes, row_count, metadata
                    FROM cache_entries
                    ORDER BY created_at DESC
                """)

            entries = []
            for row in cursor.fetchall():
                entries.append(CacheEntry(
                    key=row['key'],
                    source=row['source'],
                    series_id=row['series_id'],
                    start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else None,
                    end_date=datetime.fromisoformat(row['end_date']) if row['end_date'] else None,
                    created_at=datetime.fromisoformat(row['created_at']),
                    expires_at=datetime.fromisoformat(row['expires_at']),
                    size_bytes=row['size_bytes'],
                    row_count=row['row_count'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                ))

            return entries

    @property
    def size_mb(self) -> float:
        """Get total cache size in MB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(size_bytes) as total FROM cache_entries")
            total_bytes = cursor.fetchone()['total'] or 0
            return total_bytes / (1024 * 1024)

    @property
    def entry_count(self) -> int:
        """Get number of cache entries."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM cache_entries")
            return cursor.fetchone()['count']

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) as count,
                       SUM(size_bytes) as total_bytes,
                       SUM(row_count) as total_rows
                FROM cache_entries
            """)
            row = cursor.fetchone()

            cursor.execute("""
                SELECT source, COUNT(*) as count, SUM(size_bytes) as bytes
                FROM cache_entries
                GROUP BY source
            """)
            by_source = {r['source']: {'count': r['count'], 'bytes': r['bytes']}
                        for r in cursor.fetchall()}

            cursor.execute("""
                SELECT COUNT(*) as expired
                FROM cache_entries
                WHERE expires_at < ?
            """, (datetime.utcnow().isoformat(),))
            expired = cursor.fetchone()['expired']

            return {
                'entry_count': row['count'] or 0,
                'total_size_mb': (row['total_bytes'] or 0) / (1024 * 1024),
                'total_rows': row['total_rows'] or 0,
                'expired_entries': expired,
                'by_source': by_source,
                'db_path': str(self.db_path),
            }

    def vacuum(self) -> None:
        """Compact the database file."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")


__all__ = ['SQLiteCache', 'CacheEntry']
