"""
Audit Log Router

Exposes the tamper-evident cryptographic audit trail generated during
pipeline execution. Supports chain-integrity verification.
"""

from fastapi import APIRouter, Query
from backend.services.audit_log import get_log, get_chain_integrity, clear_log

router = APIRouter(prefix="/logs", tags=["Audit Log"])


@router.get("", summary="Get audit log entries")
@router.get("/", include_in_schema=False)
def get_audit_log(last_n: int = Query(default=100, ge=1, le=1000)):
    """Return the most recent N audit log entries with their SHA-256 chain hashes."""
    entries = get_log(last_n=last_n)
    return {
        "status":  "ok",
        "entries": entries,
        "total":   len(get_log()),
    }


@router.get("/integrity", summary="Verify chain integrity")
def verify_chain():
    """
    Recompute the full audit chain and report whether any entry has been
    tampered with. A `valid: false` result pinpoints `first_broken_seq`.
    """
    return get_chain_integrity()


@router.delete("/", summary="Clear audit log (admin)")
def clear_audit_log():
    """Wipe the in-memory audit log (admin / test operation only)."""
    clear_log()
    return {"status": "ok", "message": "Audit log cleared."}
