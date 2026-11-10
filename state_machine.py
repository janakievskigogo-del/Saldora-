"""Invoice lifecycle state machine for Saldora.

Enforces the legal lifecycle of an e-invoice at the application layer, in front
of the database:

    DRAFT  -> APPROVED  (only with human approval or an explicit override)
    APPROVED -> UBL_GENERATED
    UBL_GENERATED -> SENT
    FAILED -> DRAFT     (reset for reprocessing)
    Any state -> FAILED

Every other transition raises :class:`InvalidStateTransitionError`.  Enum values
are re-normalized on both ends so a state that came back from SQLite as a plain
string and a state passed by the caller as a string still compare cleanly.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from models import Invoice, InvoiceState

__all__ = ["InvalidStateTransitionError", "transition_invoice_state"]


class InvalidStateTransitionError(ValueError):
    """Raised when an invoice lifecycle transition is not permitted."""


# Fixed forward transitions; "any state -> FAILED" is handled explicitly below.
_ALLOWED_TRANSITIONS: frozenset[tuple[InvoiceState, InvoiceState]] = frozenset(
    {
        (InvoiceState.DRAFT, InvoiceState.APPROVED),
        (InvoiceState.APPROVED, InvoiceState.UBL_GENERATED),
        (InvoiceState.UBL_GENERATED, InvoiceState.SENT),
        (InvoiceState.FAILED, InvoiceState.DRAFT),
    }
)


def transition_invoice_state(
    invoice: Invoice,
    target_state: InvoiceState | str,
    db_session: Session | None = None,
    approve_override: bool = False,
) -> Invoice:
    """Move ``invoice`` to ``target_state``, enforcing the allowed transitions.

    Parameters
    ----------
    invoice:
        The SQLAlchemy ``models.Invoice`` instance to transition.
    target_state:
        Desired state; ``InvoiceState`` members and their string values
        (``"APPROVED"``) are both accepted and normalized.
    db_session:
        Optional session: when given, the transition is persisted
        (``add`` + ``commit``) and any database failure triggers a
        ``rollback`` before the exception is re-raised.
    approve_override:
        Human-in-the-loop escape hatch: approving a ``DRAFT`` invoice requires
        ``invoice.is_approved`` to already be ``True``; otherwise this flag must
        be set, in which case the approval stamp is also written on the invoice.

    Returns
    -------
    Invoice
        The same instance, with ``state`` (and possibly ``is_approved``) updated.

    Raises
    ------
    InvalidStateTransitionError
        For transitions outside the allowed graph, including the missing human
        approval required for ``DRAFT -> APPROVED``.
    """

    # Normalize both sides: SQLite may hand back plain strings, and callers may
    # pass enum values or strings; this makes every comparison equality-safe.
    current_state = InvoiceState(invoice.state)
    target = InvoiceState(target_state)

    if target is InvoiceState.FAILED:
        allowed = True  # Any state may fail (including FAILED itself: retry-safe no-op).
    elif current_state is InvoiceState.DRAFT and target is InvoiceState.APPROVED:
        if not (invoice.is_approved or approve_override):
            raise InvalidStateTransitionError(
                "Human approval required before approving invoice"
            )
        if approve_override:
            invoice.is_approved = True
        allowed = True
    else:
        allowed = (current_state, target) in _ALLOWED_TRANSITIONS

    if not allowed:
        raise InvalidStateTransitionError(
            f"Cannot transition from {current_state.value} to {target.value}"
        )

    invoice.state = target

    if db_session is not None:
        try:
            db_session.add(invoice)
            db_session.commit()
            # Refresh after commit so the returned instance is populated even
            # when the caller discards the session (expire_on_commit default).
            db_session.refresh(invoice)
        except Exception:
            db_session.rollback()
            raise

    return invoice
