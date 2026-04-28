#!/usr/bin/env python3
"""
Generate Ed25519 signing key pair for CIRISLens PII scrubbing.

Usage:
    python scripts/generate_scrub_key.py [--output-dir /path/to/keys]

Output:
    - lens_scrub_private.key - Private key (keep secret!)
    - lens_scrub_public.key - Public key (can be shared)
    - SQL to register the public key in the database
"""

import argparse
import base64
import os
import sys
from datetime import datetime, UTC
from pathlib import Path


def generate_keypair(output_dir: Path, key_id: str = "lens-scrub-v1"):
    """Generate Ed25519 keypair for scrub signing."""
    try:
        from nacl.signing import SigningKey
    except ImportError:
        print("Error: pynacl not installed. Run: pip install pynacl")
        sys.exit(1)

    # Generate new keypair
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key

    # Encode keys
    private_key_bytes = bytes(signing_key)
    public_key_bytes = bytes(verify_key)
    public_key_b64 = base64.b64encode(public_key_bytes).decode('ascii')

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write private key (raw bytes)
    private_key_path = output_dir / "lens_scrub_private.key"
    with open(private_key_path, 'wb') as f:
        f.write(private_key_bytes)
    os.chmod(private_key_path, 0o600)  # Owner read/write only
    print(f"Private key written to: {private_key_path}")

    # Write public key (base64)
    public_key_path = output_dir / "lens_scrub_public.key"
    with open(public_key_path, 'w') as f:
        f.write(public_key_b64)
    print(f"Public key written to: {public_key_path}")

    # Generate SQL for database registration
    sql = f"""
-- Register CIRISLens scrub signing key
INSERT INTO cirislens.lens_signing_keys (
    key_id, public_key_base64, key_type, description, created_at
) VALUES (
    '{key_id}',
    '{public_key_b64}',
    'scrub',
    'CIRISLens PII scrubbing signing key - generated {datetime.now(UTC).isoformat()}',
    NOW()
) ON CONFLICT (key_id) DO UPDATE
SET public_key_base64 = EXCLUDED.public_key_base64,
    description = EXCLUDED.description;
"""

    sql_path = output_dir / "register_scrub_key.sql"
    with open(sql_path, 'w') as f:
        f.write(sql)
    print(f"SQL written to: {sql_path}")

    print(f"\nKey ID: {key_id}")
    print(f"Public key (base64): {public_key_b64}")

    print("\n" + "=" * 60)
    print("IMPORTANT:")
    print("1. Keep lens_scrub_private.key SECRET and secure")
    print("2. Set CIRISLENS_SCRUB_KEY_PATH environment variable:")
    print(f"   export CIRISLENS_SCRUB_KEY_PATH={private_key_path.absolute()}")
    print("3. Run register_scrub_key.sql against your database")
    print("=" * 60)

    return private_key_path, public_key_path


def main():
    parser = argparse.ArgumentParser(description="Generate CIRISLens scrub signing keypair")
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("keys"),
        help="Output directory for keys (default: ./keys)"
    )
    parser.add_argument(
        "--key-id",
        type=str,
        default="lens-scrub-v1",
        help="Key ID to use (default: lens-scrub-v1)"
    )

    args = parser.parse_args()
    generate_keypair(args.output_dir, args.key_id)


if __name__ == "__main__":
    main()
