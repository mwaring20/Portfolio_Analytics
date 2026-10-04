"""
Phase 15 — Compliance Recordkeeping.

Per architecture doc:
  - Audit trail: log all analytics runs with inputs, outputs, timestamps
  - Evidence retention: store snapshots of data used in compliance decisions
  - Immutable logs: append-only storage for regulatory requirements
  - Built from day one as per design principles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Dict, List, Optional
import uuid


class EventType(Enum):
    """Types of audit events."""
    DATA_INGESTION = "Data Ingestion"
    SECURITY_RESOLUTION = "Security Resolution"
    VALUATION = "Valuation"
    CASH_FLOW_EXTRACTION = "Cash Flow Extraction"
    RETURNS_CALCULATION = "Returns Calculation"
    BENCHMARK_CONSTRUCTION = "Benchmark Construction"
    ATTRIBUTION = "Attribution"
    RISK_METRICS = "Risk Metrics"
    CORRELATION_ANALYSIS = "Correlation Analysis"
    DIVERSIFICATION_ANALYSIS = "Diversification Analysis"
    SUITABILITY_CHECK = "Suitability Check"
    STRESS_TEST = "Stress Test"
    RISK_CONTRIBUTION = "Risk Contribution"
    RECONCILIATION = "Reconciliation"


class EventSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"
    CRITICAL = "Critical"


@dataclass
class AuditEvent:
    """A single audit log event."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.INFO
    severity: EventSeverity = EventSeverity.INFO
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    account_id: Optional[str] = None
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    def __post_init__(self):
        """Compute checksum after initialization."""
        self.checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute SHA-256 checksum of event data."""
        data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "account_id": self.account_id,
            "description": self.description,
            "input_data": self._serialize_for_checksum(self.input_data),
            "output_data": self._serialize_for_checksum(self.output_data),
        }
        json_str = json.dumps(data, sort_keys=True)
        return sha256(json_str.encode()).hexdigest()

    def _serialize_for_checksum(self, data: Any) -> Any:
        """Serialize data for checksum computation."""
        if isinstance(data, dict):
            return {k: self._serialize_for_checksum(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize_for_checksum(v) for v in data]
        elif isinstance(data, Decimal):
            return str(data)
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            return data


@dataclass
class EvidenceSnapshot:
    """A snapshot of data for evidence retention."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data_type: str = ""
    data_hash: str = ""
    storage_location: str = ""
    retention_period_days: int = 2555  # 7 years default
    is_immutable: bool = True


class AuditLogger:
    """
    Audit trail logging system for compliance recordkeeping.

    Provides append-only logging and evidence retention.
    """

    def __init__(self):
        """Initialize the audit logger."""
        self._events: List[AuditEvent] = []
        self._snapshots: List[EvidenceSnapshot] = []

    def log_event(
        self,
        event_type: EventType,
        description: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        account_id: Optional[str] = None,
        severity: EventSeverity = EventSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: type of event
            description: human-readable description
            input_data: input data for the operation
            output_data: output data from the operation
            user_id: user who performed the operation
            account_id: account affected by the operation
            severity: event severity level
            metadata: additional metadata

        Returns:
            AuditEvent that was logged
        """
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            account_id=account_id,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {},
            metadata=metadata or {},
        )

        self._events.append(event)
        return event

    def get_events(
        self,
        event_type: Optional[EventType] = None,
        account_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """
        Retrieve audit events with optional filters.

        Args:
            event_type: filter by event type
            account_id: filter by account
            user_id: filter by user
            start_date: filter by start date
            end_date: filter by end date
            limit: maximum number of events to return

        Returns:
            List of matching AuditEvent objects
        """
        filtered = self._events

        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]

        if account_id:
            filtered = [e for e in filtered if e.account_id == account_id]

        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        if limit:
            filtered = filtered[-limit:]

        return filtered

    def create_evidence_snapshot(
        self,
        event_id: str,
        data: Any,
        data_type: str,
        storage_location: str,
        retention_period_days: int = 2555,
    ) -> EvidenceSnapshot:
        """
        Create an evidence snapshot for data retention.

        Args:
            event_id: associated audit event ID
            data: data to snapshot
            data_type: type of data (e.g., "positions", "returns")
            storage_location: where the data is stored
            retention_period_days: retention period in days

        Returns:
            EvidenceSnapshot that was created
        """
        # Compute hash of data
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = sha256(data_str.encode()).hexdigest()

        snapshot = EvidenceSnapshot(
            event_id=event_id,
            data_type=data_type,
            data_hash=data_hash,
            storage_location=storage_location,
            retention_period_days=retention_period_days,
        )

        self._snapshots.append(snapshot)
        return snapshot

    def get_snapshots(
        self,
        event_id: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> List[EvidenceSnapshot]:
        """
        Retrieve evidence snapshots with optional filters.

        Args:
            event_id: filter by event ID
            data_type: filter by data type

        Returns:
            List of matching EvidenceSnapshot objects
        """
        filtered = self._snapshots

        if event_id:
            filtered = [s for s in filtered if s.event_id == event_id]

        if data_type:
            filtered = [s for s in filtered if s.data_type == data_type]

        return filtered

    def verify_integrity(self) -> bool:
        """
        Verify integrity of audit log by recomputing checksums.

        Returns:
            True if all checksums are valid, False otherwise
        """
        for event in self._events:
            expected_checksum = event._compute_checksum()
            if event.checksum != expected_checksum:
                return False
        return True

    def export_audit_trail(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """
        Export audit trail as JSON string.

        Args:
            start_date: optional start date filter
            end_date: optional end date filter

        Returns:
            JSON string of audit trail
        """
        events = self.get_events(start_date=start_date, end_date=end_date)

        export_data = []
        for event in events:
            export_data.append(
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "account_id": event.account_id,
                    "description": event.description,
                    "input_data": event.input_data,
                    "output_data": event.output_data,
                    "metadata": event.metadata,
                    "checksum": event.checksum,
                }
            )

        return json.dumps(export_data, indent=2, default=str)

    def get_event_count(self) -> int:
        """Get total number of logged events."""
        return len(self._events)

    def get_snapshot_count(self) -> int:
        """Get total number of evidence snapshots."""
        return len(self._snapshots)

    def clear_events(self) -> None:
        """
        Clear all events (for testing purposes only).

        WARNING: This should never be used in production as it breaks
        the append-only requirement for compliance.
        """
        self._events.clear()
        self._snapshots.clear()


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get the global audit logger instance.

    Returns:
        Global AuditLogger instance
    """
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def log_analytics_run(
    analytics_type: EventType,
    description: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    account_id: Optional[str] = None,
) -> AuditEvent:
    """
    Convenience function to log an analytics run.

    Args:
        analytics_type: type of analytics performed
        description: description of the run
        inputs: input parameters
        outputs: output results
        account_id: account analyzed

    Returns:
        AuditEvent that was logged
    """
    logger = get_audit_logger()
    return logger.log_event(
        event_type=analytics_type,
        description=description,
        input_data=inputs,
        output_data=outputs,
        account_id=account_id,
    )
