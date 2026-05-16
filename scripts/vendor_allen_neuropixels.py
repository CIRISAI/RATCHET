#!/usr/bin/env python3
"""
Vendor a small Allen Brain Observatory Neuropixels sample for RATCHET.

Reads NWB files directly from the public anonymous S3 bucket via
fsspec + h5py (no allensdk required), extracts the drifting-gratings
spike-train block per session, and serialises to a parquet at
`data/neural/allen_neuropixels_sample.parquet` for the RATCHET A1
substrate loader.

Requires:
    pip install h5py fsspec aiohttp pyarrow

Usage:
    python3 scripts/vendor_allen_neuropixels.py --n-sessions 3 --max-units 60
    python3 scripts/vendor_allen_neuropixels.py --session-ids 715093703,719161530
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


S3_BASE = (
    "https://allen-brain-observatory.s3.amazonaws.com/"
    "visual-coding-neuropixels/ecephys-cache"
)
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "neural"


# 8 drifting-grating orientations in the Allen Visual Coding protocol
# (0, 45, ..., 315 degrees). Some trials are "blank" or have
# temporal_frequency=0; we drop those.
ORIENTATIONS = np.arange(0, 360, 45)  # 0, 45, 90, ..., 315
TEMPORAL_FREQS_KEEP = {1.0, 2.0, 4.0, 8.0, 15.0}  # standard reps


def fetch_sessions_csv() -> pd.DataFrame:
    import fsspec
    with fsspec.open(f"{S3_BASE}/sessions.csv", mode="rb") as f:
        return pd.read_csv(io.BytesIO(f.read()))


def extract_session_block(
    session_id: int,
    max_units: int = 60,
    bin_ms: float = 10.0,
    quality_filter: bool = True,
    trial_window_s: tuple = (0.0, 2.0),
    n_orientations: int = 8,
) -> dict | None:
    """Extract a drifting-grating block as a (k, t) spike-train matrix.

    Returns dict with columns matching the parquet schema in
    `ratchet.data.neural_loader._load_allen_parquet`, or None on
    failure.
    """
    import fsspec, h5py

    nwb_url = f"{S3_BASE}/session_{session_id}/session_{session_id}.nwb"
    print(f"[session {session_id}] opening {nwb_url} ...", flush=True)
    t0 = time.time()

    # Block size 1 MB is a good tradeoff: small enough for selective
    # reads, big enough that contiguous HDF5 chunks don't issue too
    # many requests.
    fobj = fsspec.open(nwb_url, mode="rb", block_size=1 * 1024 * 1024).open()
    f = h5py.File(fobj, "r")

    # ── stimulus presentations ──
    dg = f["intervals/drifting_gratings_presentations"]
    n_pres = int(dg["start_time"].shape[0])
    print(f"[session {session_id}] {n_pres} drifting-grating presentations", flush=True)

    start_time = np.asarray(dg["start_time"][:])
    stop_time = np.asarray(dg["stop_time"][:])
    orientation = np.asarray(dg["orientation"][:])  # may have NaN for blanks
    # Older NWB exports store temporal_frequency as float in [1, 2, 4, 8, 15]
    # with 0.0 sometimes meaning "blank trial"
    temporal_frequency = np.asarray(dg["temporal_frequency"][:])

    # Filter out blank/null trials
    valid = (
        ~np.isnan(orientation)
        & np.isin(np.round(temporal_frequency, 1), list(TEMPORAL_FREQS_KEEP))
    )
    if valid.sum() < 16:
        # Fallback: keep all non-NaN orientation trials regardless of TF
        valid = ~np.isnan(orientation)

    start_time = start_time[valid]
    stop_time = stop_time[valid]
    orientation = orientation[valid]

    # Map orientation degrees → discrete label in [0, n_orientations)
    label = np.round(orientation / (360.0 / n_orientations)).astype(int) % n_orientations
    n_trials = int(len(label))
    print(f"[session {session_id}] {n_trials} valid drifting-grating trials", flush=True)

    # Standard trial window: use first 2 s of presentation (or full duration)
    win_lo, win_hi = trial_window_s
    bins_per_trial = max(1, int(round((win_hi - win_lo) * 1000.0 / bin_ms)))
    n_time_bins = n_trials * bins_per_trial
    trial_bin_edges = np.arange(0, n_time_bins + 1, bins_per_trial, dtype=int)

    # ── units selection ──
    u = f["units"]
    n_units = int(u["id"].shape[0])
    print(f"[session {session_id}] {n_units} total units", flush=True)

    # Read quality + ISI violation metrics for selection
    quality = np.asarray(u["quality"][:])  # bytes objects
    isi_viol = np.asarray(u["isi_violations"][:])
    snr = np.asarray(u["snr"][:])

    quality_str = np.array([
        (q.decode() if isinstance(q, (bytes, bytearray)) else str(q))
        for q in quality
    ])

    if quality_filter:
        mask = (quality_str == "good") & (isi_viol < 0.5) & (snr > 1.0)
    else:
        mask = np.ones(n_units, dtype=bool)
    good_idx = np.flatnonzero(mask)
    if len(good_idx) == 0:
        print(f"[session {session_id}] no good units after filter; using all", flush=True)
        good_idx = np.arange(n_units)
    # Subsample to max_units
    if len(good_idx) > max_units:
        rng = np.random.default_rng(int(session_id) & 0xFFFFFFFF)
        good_idx = rng.choice(good_idx, size=max_units, replace=False)
        good_idx.sort()

    print(f"[session {session_id}] selected {len(good_idx)} units (filter='good', isi<0.5, snr>1.0)", flush=True)

    # ── spike times: read per-unit slices via spike_times_index ──
    # spike_times_index[i] = end of spikes for unit i in the flat array.
    sti = np.asarray(u["spike_times_index"][:])
    flat_spike_times = u["spike_times"]  # don't materialise — 131M floats is big

    # Build per-unit spike time arrays only for selected units
    per_unit_spike_times = {}
    for ui in good_idx:
        lo = int(sti[ui - 1]) if ui > 0 else 0
        hi = int(sti[ui])
        if hi <= lo:
            per_unit_spike_times[ui] = np.zeros(0, dtype=float)
        else:
            # Read slice from S3-backed dataset
            per_unit_spike_times[ui] = np.asarray(flat_spike_times[lo:hi])

    # ── build spike-train matrix (k, n_time_bins) ──
    k = len(good_idx)
    spike_mat = np.zeros((k, n_time_bins), dtype=np.int16)

    for trial_idx in range(n_trials):
        trial_t0 = float(start_time[trial_idx]) + win_lo
        trial_t1 = float(start_time[trial_idx]) + win_hi
        bin_lo = trial_idx * bins_per_trial
        for row_idx, unit_id in enumerate(good_idx):
            st = per_unit_spike_times[unit_id]
            # Keep spike times falling in [trial_t0, trial_t1)
            mask = (st >= trial_t0) & (st < trial_t1)
            if mask.any():
                offsets = st[mask] - trial_t0  # seconds from window start
                bin_offsets = (offsets * 1000.0 / bin_ms).astype(int)
                # Clip to [0, bins_per_trial)
                bin_offsets = bin_offsets[(bin_offsets >= 0) & (bin_offsets < bins_per_trial)]
                # Use bincount for vectorised add
                if len(bin_offsets) > 0:
                    counts = np.bincount(bin_offsets, minlength=bins_per_trial)
                    spike_mat[row_idx, bin_lo:bin_lo + bins_per_trial] += counts.astype(np.int16)

    print(f"[session {session_id}] built spike matrix shape=({k}, {n_time_bins}); total spikes={int(spike_mat.sum())}; elapsed {time.time()-t0:.1f}s",
          flush=True)

    # Encode spike_train_matrix as bytes (int16 row-major flatten)
    spike_bytes = spike_mat.astype(np.int16).tobytes()
    return {
        "session_id": f"session_{session_id}",
        "n_neurons": int(k),
        "n_trials": int(n_trials),
        "bin_ms": float(bin_ms),
        "spike_train_matrix": spike_bytes,
        "stimulus_labels": label.astype(int).tolist(),
        "trial_bin_edges": trial_bin_edges.astype(int).tolist(),
        "visual_area": "VISp",  # mixed; could refine via channel/probe lookup
        "metadata": json.dumps({
            "source": "Allen Brain Observatory Neuropixels",
            "synthetic": False,
            "trial_window_s": list(trial_window_s),
            "quality_filter": quality_filter,
            "vendoring_max_units": max_units,
            "bin_ms": float(bin_ms),
        }),
    }


def main():
    parser = argparse.ArgumentParser(description="Vendor Allen Neuropixels sample for RATCHET")
    parser.add_argument("--n-sessions", type=int, default=3, help="Number of sessions to vendor")
    parser.add_argument("--max-units", type=int, default=60, help="Max units per session")
    parser.add_argument("--session-ids", type=str, default=None,
                        help="Comma-separated session IDs (override --n-sessions)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT / "allen_neuropixels_sample.parquet"),
                        help="Output parquet path")
    parser.add_argument("--no-quality-filter", action="store_true",
                        help="Disable units quality filter (use all)")
    args = parser.parse_args()

    print("Fetching session manifest...", flush=True)
    sessions_df = fetch_sessions_csv()
    sessions_df = sessions_df[sessions_df["has_nwb"] == True].copy()
    print(f"Found {len(sessions_df)} sessions with NWB", flush=True)

    if args.session_ids:
        target_ids = [int(s.strip()) for s in args.session_ids.split(",")]
    else:
        target_ids = sessions_df["id"].head(args.n_sessions).tolist()

    rows = []
    for sid in target_ids:
        try:
            row = extract_session_block(
                int(sid),
                max_units=args.max_units,
                quality_filter=not args.no_quality_filter,
            )
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"[session {sid}] FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if not rows:
        print("No sessions extracted. Aborting.")
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB) with {len(rows)} sessions", flush=True)

    # SHA-256 pin
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"SHA-256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
