"""Tests for operator-arranged scheduled (UTC) auto approvals.

Covers creation/validation, the one-waiting-appointment rule, cancel and
reschedule, due processing approval/blocked semantics under the shared
revision arbitration, invalidation by every other approval path, waitlist
interaction, restart catch-up idempotence and history marking.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.db import SessionLocal
from app.main import (
    process_due_scheduled_approvals,
    serialize_scheduled_approval,
)
from app.models import Application, ScheduledApproval

from factories import make_payload

UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def create_app(
    client,
    *,
    start=None,
    end=None,
    circle=(0, 0, 10),
    freq=(100, 110),
    name="Event",
):
    start = start or _now() - timedelta(hours=1)
    end = end or _now() + timedelta(hours=10)
    payload = make_payload(
        circle=circle,
        freq=freq,
        start=_iso(start),
        end=_iso(end),
        name=name,
    )
    resp = client.post("/api/applications", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def schedule(client, app_id, when, *, reason="", expected=201):
    body = {"scheduled_at": _iso(when)}
    if reason:
        body["reason"] = reason
    resp = client.post(
        f"/api/applications/{app_id}/scheduled-approval", json=body
    )
    assert resp.status_code == expected, resp.text
    return resp


def current_revision(client) -> int:
    return client.get("/api/revision/current").json()["current_revision"]


# --------------------------------------------------------------------------- #
# Creation / validation
# --------------------------------------------------------------------------- #
def test_create_scheduled_approval_returns_id_and_status(client):
    app_id = create_app(client)
    when = _now() + timedelta(minutes=30)
    resp = schedule(client, app_id, when, reason="ops decision")
    body = resp.json()
    assert resp.status_code == 201
    assert body["status"] == "scheduled"
    assert isinstance(body["id"], int)
    assert body["application_id"] == app_id
    assert body["scheduled_at"] == _iso(when)
    assert body["reason"] == "ops decision"
    assert body["current_revision"] == 0
    assert "processed_at" not in body
    assert "approval_revision" not in body

    cur = client.get("/api/revision/current").json()
    # Creation consumes no revision and the application stays pending.
    assert cur["current_revision"] == 0
    assert cur["scheduled_approval_application_ids"] == [app_id]
    assert cur["scheduled_approvals"][0]["id"] == body["id"]


def test_create_rejects_unknown_application(client):
    when = _now() + timedelta(minutes=10)
    resp = schedule(client, 9999, when, expected=404)
    assert resp.json()["error"]["code"] == "not_found"
    assert client.get("/api/scheduled-approvals").json()["reservation_count"] == 0


def test_create_rejects_non_pending_application(client):
    app_id = create_app(client)
    client.post(
        f"/api/applications/{app_id}/approve",
        json={"expected_revision": 0},
    )
    when = _now() + timedelta(minutes=10)
    resp = schedule(client, app_id, when, expected=409)
    assert resp.json()["error"]["code"] == "not_pending"


def test_create_rejects_past_and_present_instant(client):
    app_id = create_app(client)
    for when in (_now() - timedelta(seconds=1), _now()):
        resp = schedule(client, app_id, when, expected=422)
        err = resp.json()["error"]
        assert err["code"] == "scheduled_at_invalid"
        codes = {d["code"] for d in err["details"]}
        assert "scheduled_at_not_in_future" in codes


def test_create_rejects_instant_at_or_after_period_end(client):
    app_row = _get_app(app_id := create_app(client))
    resp = schedule(client, app_id, _utc(app_row.end_time), expected=422)
    codes = {d["code"] for d in resp.json()["error"]["details"]}
    assert "scheduled_at_not_before_period_end" in codes

    resp2 = schedule(
        client, app_id, _utc(app_row.end_time) + timedelta(minutes=5),
        expected=422,
    )
    codes2 = {d["code"] for d in resp2.json()["error"]["details"]}
    assert "scheduled_at_not_before_period_end" in codes2


def test_create_rejects_naive_datetime(client):
    app_id = create_app(client)
    naive = (_now() + timedelta(minutes=10)).replace(tzinfo=None)
    # A naive datetime: pydantic rejects it, normalised into the standard
    # envelope.
    resp = client.post(
        f"/api/applications/{app_id}/scheduled-approval",
        json={"scheduled_at": naive.isoformat()},
    )
    assert resp.status_code == 422
    code = resp.json()["error"]["code"]
    assert code in {"validation_failed", "scheduled_at_invalid"}


def test_only_one_waiting_reservation_per_application(client):
    app_id = create_app(client)
    schedule(client, app_id, _now() + timedelta(minutes=10))
    resp = schedule(client, app_id, _now() + timedelta(minutes=20), expected=409)
    assert resp.json()["error"]["code"] == "scheduled_approval_exists"


def test_rejected_creation_leaves_no_reservation(client):
    # Past instant -> rejected, nothing created; a valid one then succeeds.
    app_id = create_app(client)
    schedule(client, app_id, _now() - timedelta(minutes=1), expected=422)
    assert (
        client.get(f"/api/applications/{app_id}/scheduled-approval")
        .json()["reservation_ids"]
        == []
    )
    schedule(client, app_id, _now() + timedelta(minutes=10))


# --------------------------------------------------------------------------- #
# Cancel / reschedule / query
# --------------------------------------------------------------------------- #
def test_cancel_before_due(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=10)).json()["id"]
    resp = client.post(f"/api/scheduled-approvals/{rid}/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    assert body["invalidated_reason"] == "cancelled"
    # Application stays pending, revision untouched.
    assert _get_app(app_id).status == "pending"
    assert current_revision(client) == 0

    # A new appointment can now be arranged.
    schedule(client, app_id, _now() + timedelta(minutes=20))
    # Cancelling twice is refused.
    resp = client.post(f"/api/scheduled-approvals/{rid}/cancel")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "scheduled_approval_not_active"


def test_cancel_unknown_reservation_404(client):
    resp = client.post("/api/scheduled-approvals/4242/cancel")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "scheduled_approval_not_found"


def test_reschedule_keeps_id_and_validates(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=10)).json()["id"]
    new_when = _now() + timedelta(minutes=45)
    resp = client.post(
        f"/api/scheduled-approvals/{rid}/reschedule",
        json={"scheduled_at": _iso(new_when), "reason": "moved"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == rid
    assert body["scheduled_at"] == _iso(new_when)
    assert body["reason"] == "moved"

    # Invalid new instant leaves the appointment unchanged.
    bad = client.post(
        f"/api/scheduled-approvals/{rid}/reschedule",
        json={"scheduled_at": _iso(_now() - timedelta(minutes=1))},
    )
    assert bad.status_code == 422
    fetched = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert fetched["scheduled_at"] == _iso(new_when)

    # Rescheduling an unknown/terminal appointment fails.
    client.post(f"/api/scheduled-approvals/{rid}/cancel")
    again = client.post(
        f"/api/scheduled-approvals/{rid}/reschedule",
        json={"scheduled_at": _iso(_now() + timedelta(minutes=5))},
    )
    assert again.status_code == 409


def test_list_and_filter_reservations(client):
    a = create_app(client)
    b = create_app(client, circle=(100, 0, 10))
    rid_a = schedule(client, a, _now() + timedelta(minutes=10)).json()["id"]
    rid_b = schedule(client, b, _now() + timedelta(minutes=20)).json()["id"]
    client.post(f"/api/scheduled-approvals/{rid_b}/cancel")

    waiting = client.get("/api/scheduled-approvals").json()
    assert waiting["reservation_ids"] == [rid_a]

    cancelled = client.get(
        "/api/scheduled-approvals", params={"status": "cancelled"}
    ).json()
    assert cancelled["reservation_ids"] == [rid_b]

    everything = client.get(
        "/api/scheduled-approvals", params={"status": "all"}
    ).json()
    assert set(everything["reservation_ids"]) == {rid_a, rid_b}

    by_app = client.get(
        "/api/scheduled-approvals",
        params={"application_id": b, "status": "all"},
    ).json()
    assert by_app["reservation_ids"] == [rid_b]

    assert (
        client.get("/api/scheduled-approvals", params={"status": "nope"})
        .status_code
        == 422
    )
    assert (
        client.get(
            "/api/scheduled-approvals", params={"application_id": 9999}
        ).status_code
        == 404
    )

    one = client.get(f"/api/scheduled-approvals/{rid_a}")
    assert one.status_code == 200
    assert one.json()["application_id"] == a
    assert client.get("/api/scheduled-approvals/9999").status_code == 404


# --------------------------------------------------------------------------- #
# Due processing
# --------------------------------------------------------------------------- #
def test_due_approval_grants_and_consumes_revision(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=5)).json()["id"]
    result = process_due_scheduled_approvals()
    # Not due yet.
    assert result["processed"] == 0
    assert _get_app(app_id).status == "pending"

    _force_due(rid)
    result = process_due_scheduled_approvals()
    assert result["approved_reservation_ids"] == [rid]

    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["status"] == "approved"
    assert body["approval_revision"] == 1
    assert "processed_at" in body

    app = client.get(f"/api/applications/{app_id}").json()
    assert app["status"] == "approved"
    assert app["approval_revision"] == 1

    snap = client.get("/api/revision/1").json()
    record = next(a for a in snap["approvals"] if a["revision"] == 1)
    assert record["scheduled_approved"] is True
    assert record["scheduled_reservation_id"] == rid
    assert record["auto_approved"] is False
    assert record["emergency"] is False


def test_due_blocked_records_detail_no_revision_pending(client):
    # A first effective license occupies the same spectrum.
    owner = create_app(client, freq=(100, 110))
    client.post(
        f"/api/applications/{owner}/approve",
        json={"expected_revision": 0},
    )
    victim = create_app(client, freq=(105, 115))
    rid = schedule(client, victim, _now() + timedelta(minutes=5)).json()["id"]
    _force_due(rid)

    result = process_due_scheduled_approvals()
    assert result["blocked_reservation_ids"] == [rid]

    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["status"] == "blocked"
    assert "approval_revision" not in body
    details = body["blocked_details"]
    assert details["conflicts"]
    assert details["conflicts"][0]["application"]["id"] == owner
    assert details["current_revision"] == 1

    # Application still pending, no revision consumed.
    assert _get_app(victim).status == "pending"
    assert current_revision(client) == 1

    # Blocked appointments are terminal: a later tick must not retry, even
    # after the blocker is revoked.
    client.post(f"/api/applications/{owner}/revoke", json={"reason": "gone"})
    assert process_due_scheduled_approvals()["processed"] == 0
    assert _get_app(victim).status == "pending"
    assert client.get(f"/api/scheduled-approvals/{rid}").json()["status"] == "blocked"


def test_due_blocked_by_restriction_window(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=5)).json()["id"]
    # Activate a window covering the whole candidate spectrum/time/area.
    rev = current_revision(client)
    client.post(
        "/api/restriction-windows",
        json={
            "circle": {"cx": 0, "cy": 0, "radius": 100},
            "frequency": {"low": 90, "high": 120},
            "time_window": {
                "start": _iso(_now() - timedelta(hours=1)),
                "end": _iso(_now() + timedelta(hours=10)),
            },
            "reason": "control",
            "expected_revision": rev,
        },
    )
    _force_due(rid)
    result = process_due_scheduled_approvals()
    assert result["blocked_reservation_ids"] == [rid]
    details = client.get(f"/api/scheduled-approvals/{rid}").json()[
        "blocked_details"
    ]
    assert details["restriction_window_conflicts"]
    assert not details["conflicts"]
    assert _get_app(app_id).status == "pending"


def test_due_processing_uses_live_guard_band(client):
    # Appointments are adjudicated with the guard band in force at processing
    # time, not arrangement time: tighten the band to 0 so a 1 MHz gap passes.
    owner = create_app(client, freq=(100, 110))
    client.post(
        f"/api/applications/{owner}/approve",
        json={"expected_revision": 0},
    )
    rev = current_revision(client)
    client.post(
        "/api/config/guard-band",
        json={"guard_band": 0.0, "expected_revision": rev},
    )
    victim = create_app(client, circle=(0, 0, 10), freq=(111, 112))
    rid = schedule(client, victim, _now() + timedelta(minutes=5)).json()["id"]
    _force_due(rid)
    result = process_due_scheduled_approvals()
    assert result["approved_reservation_ids"] == [rid]


def test_multiple_due_processed_in_appointment_order(client):
    # Two non-conflicting pending applications with past appointments.
    a = create_app(client, circle=(0, 0, 10), freq=(100, 110))
    b = create_app(client, circle=(5000, 0, 10), freq=(100, 110))
    rid_a = schedule(client, a, _now() + timedelta(minutes=5)).json()["id"]
    rid_b = schedule(client, b, _now() + timedelta(minutes=6)).json()["id"]
    _force_due(rid_a)
    _force_due(rid_b)
    result = process_due_scheduled_approvals()
    assert result["approved_reservation_ids"] == [rid_a, rid_b]
    assert client.get(f"/api/scheduled-approvals/{rid_a}").json()[
        "approval_revision"
    ] == 1
    assert client.get(f"/api/scheduled-approvals/{rid_b}").json()[
        "approval_revision"
    ] == 2


def test_blocked_due_does_not_stop_later_appointments(client):
    # A blocked appointment is skipped; a later, free appointment still fires.
    owner = create_app(client, freq=(100, 110))
    client.post(
        f"/api/applications/{owner}/approve",
        json={"expected_revision": 0},
    )
    blocked_app = create_app(client, freq=(105, 115))
    free_app = create_app(client, circle=(5000, 0, 10), freq=(100, 110))
    rid_blocked = schedule(
        client, blocked_app, _now() + timedelta(minutes=5)
    ).json()["id"]
    rid_free = schedule(
        client, free_app, _now() + timedelta(minutes=6)
    ).json()["id"]
    _force_due(rid_blocked)
    _force_due(rid_free)
    result = process_due_scheduled_approvals()
    assert result["blocked_reservation_ids"] == [rid_blocked]
    assert result["approved_reservation_ids"] == [rid_free]


def test_scheduled_approval_removes_waitlist_without_draining(client):
    owner = create_app(client, freq=(100, 110))
    client.post(
        f"/api/applications/{owner}/approve",
        json={"expected_revision": 0},
    )
    # Two queued pending apps; the first one also has a scheduled appointment.
    q1 = create_app(client, freq=(105, 115))
    q2 = create_app(client, freq=(106, 116))
    client.post(f"/api/applications/{q1}/waitlist", json={"reason": "q1"})
    client.post(f"/api/applications/{q2}/waitlist", json={"reason": "q2"})
    rid = schedule(client, q1, _now() + timedelta(minutes=5)).json()["id"]

    # The owner is revoked: the queue drain would (with the guard band 2.0)
    # still block both, so first give q1's scheduled approval a chance after
    # freeing the spectrum differently: revoke owner -> drain blocks q1/q2
    # (they clash with each other? no -- they clash mutually in the same
    # spectrum). Make the scenario deterministic instead: revoke the owner
    # and move q2 away before the appointment fires.
    client.post(f"/api/applications/{q2}/waitlist/cancel")
    client.post(f"/api/applications/{owner}/revoke", json={"reason": "free"})
    # The drain just auto-approved q1 at the release revision: its scheduled
    # appointment must be invalidated, not fired.
    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["status"] == "invalidated"
    assert body["invalidated_reason"] == "waitlist_auto_approval"
    assert _get_app(q1).status == "approved"
    # Idempotent: a later tick changes nothing.
    assert process_due_scheduled_approvals()["processed"] == 0


def test_scheduled_approval_of_waitlisted_app_passes_without_draining(client):
    # A queued application whose appointment becomes due while the spectrum is
    # free leaves the queue and gets approved WITHOUT triggering a waitlist
    # drain.  Setup: the drain at release time keeps q1 queued because an
    # active restriction window blocks it; the window is then lifted (which
    # itself never drains the queue) and the appointment fires.
    owner = create_app(client, freq=(100, 110))
    client.post(
        f"/api/applications/{owner}/approve",
        json={"expected_revision": 0},
    )
    q1 = create_app(client, freq=(100, 110))
    client.post(f"/api/applications/{q1}/waitlist")
    rid = schedule(client, q1, _now() + timedelta(minutes=5)).json()["id"]

    rev = current_revision(client)
    win = client.post(
        "/api/restriction-windows",
        json={
            "circle": {"cx": 0, "cy": 0, "radius": 100},
            "frequency": {"low": 100, "high": 110},
            "time_window": {
                "start": _iso(_now() - timedelta(hours=1)),
                "end": _iso(_now() + timedelta(hours=10)),
            },
            "reason": "block drain",
            "expected_revision": rev,
        },
    ).json()["id"]
    # Revoke the owner: the drain re-adjudicates q1, the window blocks it, so
    # q1 stays queued and pending (the revoke consumes exactly one revision).
    client.post(f"/api/applications/{owner}/revoke", json={"reason": "free"})
    assert _get_app(q1).status == "pending"
    assert client.get(f"/api/applications/{q1}/waitlist").status_code == 200
    # Window activation consumed rev+1; the blocked drain consumed no further
    # revision, so the revoke itself landed at rev+2.
    rev_after_revoke = current_revision(client)
    assert rev_after_revoke == rev + 2

    # Lift the window (one revision, no drain), then the appointment fires and
    # approves q1; q1 leaves the queue and no other drain is triggered.
    client.post(
        f"/api/restriction-windows/{win}/release",
        json={"expected_revision": rev_after_revoke},
    )
    _force_due(rid)
    result = process_due_scheduled_approvals()
    assert result["approved_reservation_ids"] == [rid]
    assert _get_app(q1).status == "approved"
    # Queue entry removed.
    assert client.get(f"/api/applications/{q1}/waitlist").status_code == 404
    # Window release consumes one revision; the scheduled approval then the
    # next one (release=4, scheduled approval=5 with owner approval at 1).
    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["approval_revision"] == rev_after_revoke + 2


# --------------------------------------------------------------------------- #
# Invalidation by other approval paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind",
    ["manual", "batch", "transfer", "emergency", "waitlist"],
)
def test_approval_paths_invalidate_waiting_reservation(client, kind):
    target = create_app(client, freq=(100, 110))
    rid = schedule(client, target, _now() + timedelta(minutes=5)).json()["id"]
    rev = current_revision(client)

    if kind == "manual":
        resp = client.post(
            f"/api/applications/{target}/approve",
            json={"expected_revision": rev},
        )
        assert resp.status_code == 200
    elif kind == "batch":
        resp = client.post(
            "/api/applications/batch-approve",
            json={"application_ids": [target], "expected_revision": rev},
        )
        assert resp.status_code == 200
    elif kind == "transfer":
        # Handover: need an effective source to swap out.
        source = create_app(
            client, circle=(0, 0, 10), freq=(100, 110),
            start=_now() - timedelta(hours=1),
            end=_now() + timedelta(hours=10),
        )
        client.post(
            f"/api/applications/{source}/approve",
            json={"expected_revision": rev},
        )
        # The successor must describe spectrum free once the source is gone;
        # recreate target's band equal to the source (identical geometry).
        rev2 = current_revision(client)
        resp = client.post(
            "/api/license-transfers",
            json={
                "source_application_id": source,
                "successor_application_id": target,
                "expected_revision": rev2,
                "reason": "swap",
            },
        )
        assert resp.status_code == 200, resp.text
    elif kind == "emergency":
        resp = client.post(
            "/api/emergency-occupations",
            json={"application_id": target, "expected_revision": rev},
        )
        assert resp.status_code == 201, resp.text
    elif kind == "waitlist":
        # An effective owner blocks the target; queue it, then release the
        # owner to trigger the FIFO drain auto approval.
        owner = create_app(client, freq=(100, 110))
        client.post(
            f"/api/applications/{owner}/approve",
            json={"expected_revision": rev},
        )
        client.post(f"/api/applications/{target}/waitlist")
        client.post(f"/api/applications/{owner}/revoke", json={"reason": "x"})

    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["status"] == "invalidated"
    expected_reason = {
        "manual": "manual_approval",
        "batch": "batch_approval",
        "transfer": "license_transfer",
        "emergency": "emergency_approval",
        "waitlist": "waitlist_auto_approval",
    }[kind]
    assert body["invalidated_reason"] == expected_reason
    # A due tick after invalidation does nothing.
    _force_due(rid)
    assert process_due_scheduled_approvals()["processed"] == 0


# --------------------------------------------------------------------------- #
# Restart / concurrency / history
# --------------------------------------------------------------------------- #
def test_restart_catchup_processes_due_once(client):
    a = create_app(client, circle=(0, 0, 10), freq=(100, 110))
    b = create_app(client, circle=(5000, 0, 10), freq=(100, 110))
    rid_a = schedule(client, a, _now() + timedelta(minutes=5)).json()["id"]
    rid_b = schedule(client, b, _now() + timedelta(minutes=6)).json()["id"]
    _force_due(rid_a)
    _force_due(rid_b)

    # First catch-up approves both with revisions 1 and 2.
    first = process_due_scheduled_approvals()
    assert first["approved_reservation_ids"] == [rid_a, rid_b]
    # Second catch-up (simulating another restart) must not approve again.
    second = process_due_scheduled_approvals()
    assert second["processed"] == 0
    assert _get_app(a).approval_revision == 1
    assert _get_app(b).approval_revision == 2
    # Exactly two approval events.
    assert len(client.get("/api/revision/2").json()["approvals"]) == 2


def test_manual_approval_racing_due_processing_wins_once(client):
    # The appointment is due; a manual approval committed first must leave the
    # appointment invalidated and must not produce a second approval.
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=5)).json()["id"]
    client.post(
        f"/api/applications/{app_id}/approve",
        json={"expected_revision": 0},
    )
    _force_due(rid)
    result = process_due_scheduled_approvals()
    assert result["processed"] == 0
    body = client.get(f"/api/scheduled-approvals/{rid}").json()
    assert body["status"] == "invalidated"
    assert body["invalidated_reason"] == "manual_approval"
    assert _get_app(app_id).approval_revision == 1


def test_history_marks_scheduled_approval_and_keeps_event_semantics(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=5)).json()["id"]
    _force_due(rid)
    process_due_scheduled_approvals()
    cur = current_revision(client)
    snap = client.get(f"/api/revision/{cur}").json()
    approval = snap["approvals"][0]
    assert approval["revision"] == 1
    assert approval["application_id"] == app_id
    assert approval["scheduled_approved"] is True
    assert approval["scheduled_reservation_id"] == rid
    assert approval["status_as_of"] == "approved"
    assert snap["effective_application_ids"] == [app_id]


def test_serialization_covers_terminal_fields(client):
    app_id = create_app(client)
    rid = schedule(client, app_id, _now() + timedelta(minutes=5)).json()["id"]
    client.post(f"/api/scheduled-approvals/{rid}/cancel")
    with SessionLocal() as session:
        row = session.get(ScheduledApproval, rid)
        body = serialize_scheduled_approval(session, row)
    assert body["status"] == "cancelled"
    assert body["invalidated_reason"] == "cancelled"
    assert "invalidated_at" in body


def test_background_worker_fires_due_appointment(client):
    # End-to-end through the live poller: an appointment created 1 second in
    # the future is auto-approved without any manual processing call.
    import time as _time

    import app.main as main_module

    worker = main_module.scheduled_approval_worker
    assert worker is not None
    worker.restart(0.05)
    app_id = create_app(client)
    schedule(client, app_id, _now() + timedelta(seconds=1))
    deadline = _time.monotonic() + 10.0
    while _time.monotonic() < deadline:
        if _get_app(app_id).status == "approved":
            break
        _time.sleep(0.05)
    assert _get_app(app_id).status == "approved"
    assert _get_app(app_id).approval_revision == 1
    worker.restart(3600)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _get_app(app_id: int) -> Application:
    with SessionLocal() as session:
        row = session.get(Application, app_id)
        session.expunge(row)
        return row


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _force_due(rid: int, when: datetime | None = None) -> None:
    """Move a waiting appointment's instant into the past (simulated time)."""
    instant = when or (_now() - timedelta(seconds=1))
    with SessionLocal() as session, session.begin():
        resv = session.get(ScheduledApproval, rid)
        resv.scheduled_at = instant.astimezone(timezone.utc)
