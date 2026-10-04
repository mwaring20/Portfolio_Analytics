"""
Tests for Phase 15 Compliance Recordkeeping.
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from pillar1.compliance import (
    AuditEvent,
    AuditLogger,
    EvidenceSnapshot,
    EventSeverity,
    EventType,
    get_audit_logger,
    log_analytics_run,
)


def test_log_event():
    """Test logging a basic audit event."""
    logger = AuditLogger()

    event = logger.log_event(
        event_type=EventType.VALUATION,
        description="Portfolio valuation run",
        input_data={"account_id": "ACC001", "date": "2024-01-31"},
        output_data={"total_value": 10000},
        user_id="user123",
        account_id="ACC001",
    )

    assert event.event_type == EventType.VALUATION
    assert event.description == "Portfolio valuation run"
    assert event.user_id == "user123"
    assert event.account_id == "ACC001"
    assert event.checksum is not None


def test_log_event_with_severity():
    """Test logging event with severity level."""
    logger = AuditLogger()

    event = logger.log_event(
        event_type=EventType.SUITABILITY_CHECK,
        description="Suitability violation detected",
        severity=EventSeverity.WARNING,
    )

    assert event.severity == EventSeverity.WARNING


def test_get_events_no_filter():
    """Test retrieving all events without filters."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "Valuation 1")
    logger.log_event(EventType.RETURNS_CALCULATION, "Returns 1")
    logger.log_event(EventType.VALUATION, "Valuation 2")

    events = logger.get_events()

    assert len(events) == 3


def test_get_events_with_type_filter():
    """Test retrieving events filtered by type."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "Valuation 1")
    logger.log_event(EventType.RETURNS_CALCULATION, "Returns 1")
    logger.log_event(EventType.VALUATION, "Valuation 2")

    events = logger.get_events(event_type=EventType.VALUATION)

    assert len(events) == 2
    assert all(e.event_type == EventType.VALUATION for e in events)


def test_get_events_with_account_filter():
    """Test retrieving events filtered by account."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "Valuation 1", account_id="ACC001")
    logger.log_event(EventType.VALUATION, "Valuation 2", account_id="ACC002")
    logger.log_event(EventType.VALUATION, "Valuation 3", account_id="ACC001")

    events = logger.get_events(account_id="ACC001")

    assert len(events) == 2
    assert all(e.account_id == "ACC001" for e in events)


def test_get_events_with_date_filter():
    """Test retrieving events filtered by date range."""
    logger = AuditLogger()

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    # Manually set timestamps for testing
    event1 = AuditEvent(
        event_type=EventType.VALUATION,
        description="Old event",
        timestamp=yesterday,
    )
    event2 = AuditEvent(
        event_type=EventType.VALUATION,
        description="New event",
        timestamp=now,
    )

    logger._events = [event1, event2]

    events = logger.get_events(start_date=now - timedelta(hours=1))

    assert len(events) == 1
    assert events[0].description == "New event"


def test_get_events_with_limit():
    """Test retrieving events with limit."""
    logger = AuditLogger()

    for i in range(10):
        logger.log_event(EventType.VALUATION, f"Event {i}")

    events = logger.get_events(limit=5)

    assert len(events) == 5


def test_create_evidence_snapshot():
    """Test creating an evidence snapshot."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    data = {"positions": [{"security_id": "VTI", "quantity": 10}]}

    snapshot = logger.create_evidence_snapshot(
        event_id=event.event_id,
        data=data,
        data_type="positions",
        storage_location="/snapshots/positions_001.json",
    )

    assert snapshot.event_id == event.event_id
    assert snapshot.data_type == "positions"
    assert snapshot.data_hash is not None
    assert snapshot.storage_location == "/snapshots/positions_001.json"


def test_create_evidence_snapshot_with_retention():
    """Test creating snapshot with custom retention period."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    snapshot = logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"test": "data"},
        data_type="test",
        storage_location="/snapshots/test.json",
        retention_period_days=3650,  # 10 years
    )

    assert snapshot.retention_period_days == 3650


def test_get_snapshots():
    """Test retrieving snapshots."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"type": "positions"},
        data_type="positions",
        storage_location="/snapshots/pos.json",
    )

    logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"type": "returns"},
        data_type="returns",
        storage_location="/snapshots/ret.json",
    )

    snapshots = logger.get_snapshots()

    assert len(snapshots) == 2


def test_get_snapshots_with_filter():
    """Test retrieving snapshots filtered by data type."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"type": "positions"},
        data_type="positions",
        storage_location="/snapshots/pos.json",
    )

    logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"type": "returns"},
        data_type="returns",
        storage_location="/snapshots/ret.json",
    )

    snapshots = logger.get_snapshots(data_type="positions")

    assert len(snapshots) == 1
    assert snapshots[0].data_type == "positions"


def test_verify_integrity():
    """Test integrity verification of audit log."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "Valuation")
    logger.log_event(EventType.RETURNS_CALCULATION, "Returns")

    assert logger.verify_integrity() is True


def test_verify_integrity_tampered():
    """Test integrity verification detects tampering."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    # Tamper with the event
    event.description = "Tampered description"

    assert logger.verify_integrity() is False


def test_export_audit_trail():
    """Test exporting audit trail as JSON."""
    logger = AuditLogger()

    logger.log_event(
        EventType.VALUATION,
        "Valuation",
        input_data={"account": "ACC001"},
        output_data={"value": 10000},
    )

    export = logger.export_audit_trail()

    assert "Valuation" in export
    assert "ACC001" in export
    assert "10000" in export


def test_export_audit_trail_with_date_filter():
    """Test exporting audit trail with date filter."""
    logger = AuditLogger()

    now = datetime.utcnow()
    yesterday = now - timedelta(days=1)

    event1 = AuditEvent(
        event_type=EventType.VALUATION,
        description="Old event",
        timestamp=yesterday,
    )
    event2 = AuditEvent(
        event_type=EventType.VALUATION,
        description="New event",
        timestamp=now,
    )

    logger._events = [event1, event2]

    export = logger.export_audit_trail(start_date=now - timedelta(hours=1))

    assert "New event" in export
    assert "Old event" not in export


def test_get_event_count():
    """Test getting event count."""
    logger = AuditLogger()

    assert logger.get_event_count() == 0

    logger.log_event(EventType.VALUATION, "Valuation 1")
    logger.log_event(EventType.VALUATION, "Valuation 2")

    assert logger.get_event_count() == 2


def test_get_snapshot_count():
    """Test getting snapshot count."""
    logger = AuditLogger()

    event = logger.log_event(EventType.VALUATION, "Valuation")

    assert logger.get_snapshot_count() == 0

    logger.create_evidence_snapshot(
        event_id=event.event_id,
        data={"test": "data"},
        data_type="test",
        storage_location="/snapshots/test.json",
    )

    assert logger.get_snapshot_count() == 1


def test_clear_events():
    """Test clearing events (for testing)."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "Valuation")
    logger.log_event(EventType.RETURNS_CALCULATION, "Returns")

    assert logger.get_event_count() == 2

    logger.clear_events()

    assert logger.get_event_count() == 0
    assert logger.get_snapshot_count() == 0


def test_global_audit_logger():
    """Test global audit logger singleton."""
    logger1 = get_audit_logger()
    logger2 = get_audit_logger()

    assert logger1 is logger2


def test_log_analytics_run_convenience():
    """Test convenience function for logging analytics runs."""
    event = log_analytics_run(
        analytics_type=EventType.RISK_METRICS,
        description="Risk metrics calculation",
        inputs={"account_id": "ACC001", "date": "2024-01-31"},
        outputs={"volatility": 0.15, "sharpe": 1.5},
        account_id="ACC001",
    )

    assert event.event_type == EventType.RISK_METRICS
    assert event.account_id == "ACC001"
    assert event.input_data["account_id"] == "ACC001"
    assert event.output_data["volatility"] == 0.15


def test_audit_event_checksum_computation():
    """Test that checksum is computed correctly."""
    event = AuditEvent(
        event_type=EventType.VALUATION,
        description="Test event",
        user_id="user123",
        account_id="ACC001",
        input_data={"key": "value"},
        output_data={"result": 42},
    )

    assert event.checksum is not None
    assert len(event.checksum) == 64  # SHA-256 produces 64 hex characters


def test_audit_event_checksum_serialization():
    """Test that Decimal and datetime are serialized correctly for checksum."""
    event = AuditEvent(
        event_type=EventType.VALUATION,
        description="Test",
        input_data={"amount": Decimal("100.50")},
        output_data={"timestamp": datetime(2024, 1, 31, 12, 0)},
    )

    # Should not raise an error
    checksum = event._compute_checksum()
    assert checksum is not None


def test_event_type_enum():
    """Test EventType enum values."""
    assert EventType.DATA_INGESTION.value == "Data Ingestion"
    assert EventType.VALUATION.value == "Valuation"
    assert EventType.SUITABILITY_CHECK.value == "Suitability Check"


def test_event_severity_enum():
    """Test EventSeverity enum values."""
    assert EventSeverity.INFO.value == "Info"
    assert EventSeverity.WARNING.value == "Warning"
    assert EventSeverity.ERROR.value == "Error"
    assert EventSeverity.CRITICAL.value == "Critical"


def test_evidence_snapshot_dataclass():
    """Test EvidenceSnapshot dataclass structure."""
    snapshot = EvidenceSnapshot(
        event_id="event123",
        data_type="positions",
        data_hash="abc123",
        storage_location="/snapshots/pos.json",
        retention_period_days=2555,
        is_immutable=True,
    )

    assert snapshot.event_id == "event123"
    assert snapshot.data_type == "positions"
    assert snapshot.is_immutable is True


def test_audit_event_dataclass_defaults():
    """Test AuditEvent dataclass default values."""
    event = AuditEvent(
        event_type=EventType.VALUATION,
        description="Test",
    )

    assert event.event_id is not None  # UUID generated
    assert event.severity == EventSeverity.INFO  # Default
    assert event.user_id is None  # Default
    assert event.account_id is None  # Default
    assert event.input_data == {}  # Default
    assert event.output_data == {}  # Default
    assert event.metadata == {}  # Default


def test_multiple_filters_combined():
    """Test using multiple filters together."""
    logger = AuditLogger()

    logger.log_event(EventType.VALUATION, "V1", account_id="ACC001", user_id="user1")
    logger.log_event(EventType.VALUATION, "V2", account_id="ACC002", user_id="user1")
    logger.log_event(EventType.RETURNS_CALCULATION, "R1", account_id="ACC001", user_id="user2")

    events = logger.get_events(
        event_type=EventType.VALUATION,
        account_id="ACC001",
        user_id="user1",
    )

    assert len(events) == 1
    assert events[0].description == "V1"


def test_log_event_with_metadata():
    """Test logging event with metadata."""
    logger = AuditLogger()

    event = logger.log_event(
        event_type=EventType.VALUATION,
        description="Valuation",
        metadata={"version": "1.0", "runtime_ms": 150},
    )

    assert event.metadata["version"] == "1.0"
    assert event.metadata["runtime_ms"] == 150
