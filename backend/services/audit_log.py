"""
Cryptographic Audit Log Service

Maintains a tamper-evident, blockchain-style chain of log entries using SHA-256.
Each entry includes the hash of the previous entry, making any tampering detectable.

References:
  NIST SP 800-92 (2006) Guide to Computer Security Log Management.
  ISO/IEC 27037:2012 Digital Evidence Guidelines.
"""

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


_audit_log: List[Dict[str, Any]] = []
_log_lock = threading.Lock()


def _sha256_of(data: Any) -> str:
    """Deterministic SHA-256 of JSON-serializable data."""
    raw = json.dumps(data, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def add_entry(
    message: str,
    module: str = "SYSTEM",
    level: str = "INFO",
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Append a tamper-evident audit entry to the chain.

    The `entry_hash` covers all fields including `prev_hash`, so any
    post-hoc modification of any field or insertion between entries
    will break the chain and be detectable via verify_chain().
    """
    with _log_lock:
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_hash = _audit_log[-1]["entry_hash"] if _audit_log else "genesis"
        seq = len(_audit_log)

        payload: Dict[str, Any] = {
            "seq":       seq,
            "timestamp": timestamp,
            "level":     level,
            "module":    module,
            "message":   message,
            "prev_hash": prev_hash,
        }
        if metadata:
            payload["metadata"] = {k: str(v)[:300] for k, v in metadata.items()}

        payload["entry_hash"] = _sha256_of(payload)
        _audit_log.append(payload)
        return dict(payload)


def get_log(last_n: Optional[int] = None) -> List[Dict]:
    """Return audit log entries (most recent `last_n`, or all)."""
    with _log_lock:
        if last_n is not None:
            return list(_audit_log[-last_n:])
        return list(_audit_log)


def get_chain_integrity() -> Dict:
    """
    Verify the cryptographic chain integrity of the entire audit log.

    Returns
    -------
    dict with `valid`, `n_entries`, `chain_head`, and `first_broken_seq`.
    """
    with _log_lock:
        if not _audit_log:
            return {"valid": True, "n_entries": 0, "chain_head": "genesis"}

        prev_hash = "genesis"
        first_broken = None

        for entry in _audit_log:
            # 1. Verify linkage
            if entry.get("prev_hash") != prev_hash:
                first_broken = entry["seq"]
                break

            # 2. Verify self-hash (recompute without the entry_hash field)
            entry_copy = {k: v for k, v in entry.items() if k != "entry_hash"}
            computed = _sha256_of(entry_copy)
            if computed != entry.get("entry_hash"):
                first_broken = entry["seq"]
                break

            prev_hash = entry["entry_hash"]

        return {
            "valid":             first_broken is None,
            "n_entries":         len(_audit_log),
            "chain_head":        _audit_log[-1]["entry_hash"],
            "chain_root":        _audit_log[0]["entry_hash"] if _audit_log else "genesis",
            "first_broken_seq":  first_broken,
        }


def clear_log() -> None:
    """Clear all entries (use only in tests / admin resets)."""
    with _log_lock:
        _audit_log.clear()
