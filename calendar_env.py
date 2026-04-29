import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import gymnasium as gym
from gymnasium import spaces

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:  # pragma: no cover - allows simulated mode without Google libs
    Credentials = None
    build = None
    HttpError = Exception


def _now_utc_iso() -> str:
    return datetime.now(tz=ZoneInfo("UTC")).isoformat()


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _parse_iso(ts: str) -> datetime:
    normalized = ts.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include timezone: {ts}")
    return parsed


class CalendarSchedulingEnv(gym.Env):
    """RL-style scheduling environment for Google Calendar or simulated backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        calendar_id: Optional[str] = None,
        credentials_path: Optional[str] = None,
        backend: str = "simulated",
        timezone: str = "America/New_York",
        max_steps: int = 25,
        work_start_hour: int = 9,
        work_end_hour: int = 18,
        managed_prefix: str = "[RLGYM]",
        read_backoff_seconds: float = 1.0,
    ) -> None:
        super().__init__()

        if backend not in {"simulated", "live"}:
            raise ValueError("backend must be either 'simulated' or 'live'")
        if backend == "live" and (not calendar_id or not credentials_path):
            raise ValueError("calendar_id and credentials_path are required in live backend")

        self.backend = backend
        self.calendar_id = calendar_id
        self.credentials_path = credentials_path
        self.tz = ZoneInfo(timezone)
        self.max_steps = int(max_steps)
        self.work_start_hour = int(work_start_hour)
        self.work_end_hour = int(work_end_hour)
        self.managed_prefix = managed_prefix
        self.read_backoff_seconds = float(read_backoff_seconds)

        self.action_space = spaces.Dict(
            {
                "type": spaces.Text(min_length=0, max_length=32),
                "event_id": spaces.Text(min_length=0, max_length=128),
                "request_id": spaces.Text(min_length=0, max_length=128),
                "start_iso": spaces.Text(min_length=0, max_length=64),
                "end_iso": spaces.Text(min_length=0, max_length=64),
                "title": spaces.Text(min_length=0, max_length=256),
                "payload_json": spaces.Text(min_length=0, max_length=4096),
            }
        )
        self.observation_space = spaces.Text(min_length=0, max_length=50000)

        self.current_step = 0
        self.episode_id = ""
        self.day_anchor: datetime = datetime.now(tz=self.tz)
        self.scenario_name = ""
        self.events: List[Dict[str, Any]] = []
        self.requests: List[Dict[str, Any]] = []
        self.completed_requests: Dict[str, str] = {}
        self.last_error: Optional[str] = None
        self.last_action: Dict[str, str] = {}
        self._rewarded_requests: Dict[str, bool] = {}
        self._service = None

        if self.backend == "live":
            self._service = self._build_live_service()

    def _build_live_service(self):
        if Credentials is None or build is None:
            raise RuntimeError("Google API packages are unavailable. Install requirements first.")
        scopes = ["https://www.googleapis.com/auth/calendar"]
        credentials = Credentials.from_service_account_file(
            self.credentials_path,
            scopes=scopes,
        )
        return build("calendar", "v3", credentials=credentials, cache_discovery=False)

    def _execute_with_retries(
        self,
        fn,
        operation_name: str,
        ignore_http_statuses: Optional[set] = None,
    ):
        max_attempts = 6
        base_backoff = 1.8
        ignored = ignore_http_statuses or set()
        for attempt in range(1, max_attempts + 1):
            try:
                return fn()
            except HttpError as exc:  # type: ignore[misc]
                status = getattr(getattr(exc, "resp", None), "status", None)
                if status in ignored:
                    return None
                retriable = status in {429, 500, 502, 503, 504}
                if retriable and attempt < max_attempts:
                    time.sleep(base_backoff ** attempt)
                    continue
                raise RuntimeError(f"Calendar API error during {operation_name}: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Unexpected error during {operation_name}: {exc}") from exc

    def _scenario_templates(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "simple_back_to_back",
                "seed_events": [
                    {"title": "Focus", "start_min": 10 * 60, "end_min": 11 * 60, "locked": True, "priority": 1},
                    {"title": "Team Sync", "start_min": 13 * 60, "end_min": 14 * 60, "locked": True, "priority": 2},
                ],
                "requests": [
                    {"request_id": "R1", "title": "Client Call", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 3, "must_schedule": True},
                    {"request_id": "R2", "title": "Budget Review", "duration_min": 45, "earliest_min": 11 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": True},
                ],
            },
            {
                "name": "high_priority_collision",
                "seed_events": [
                    {"title": "Planning", "start_min": 11 * 60, "end_min": 12 * 60, "locked": False, "priority": 1},
                    {"title": "Design", "start_min": 14 * 60, "end_min": 15 * 60, "locked": True, "priority": 2},
                ],
                "requests": [
                    {"request_id": "R3", "title": "Exec Escalation", "duration_min": 60, "earliest_min": 11 * 60, "latest_min": 15 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R4", "title": "Vendor Check", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 16 * 60, "priority": 2, "must_schedule": False},
                ],
            },
            {
                "name": "tight_workday",
                "seed_events": [
                    {"title": "Daily Standup", "start_min": 9 * 60, "end_min": 9 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Deep Work", "start_min": 10 * 60, "end_min": 12 * 60, "locked": True, "priority": 2},
                    {"title": "Customer Demo", "start_min": 15 * 60, "end_min": 16 * 60, "locked": True, "priority": 4},
                ],
                "requests": [
                    {"request_id": "R5", "title": "Incident Review", "duration_min": 60, "earliest_min": 12 * 60, "latest_min": 17 * 60, "priority": 4, "must_schedule": True},
                    {"request_id": "R6", "title": "1:1", "duration_min": 30, "earliest_min": 13 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": True},
                    {"request_id": "R7", "title": "Docs", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 1, "must_schedule": False},
                ],
            },
            {
                "name": "meeting_overload",
                "seed_events": [
                    {"title": "Blocked", "start_min": 9 * 60, "end_min": 11 * 60, "locked": True, "priority": 1},
                    {"title": "Blocked", "start_min": 12 * 60, "end_min": 14 * 60, "locked": True, "priority": 1},
                    {"title": "Blocked", "start_min": 15 * 60, "end_min": 17 * 60, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R8", "title": "Urgent Fix", "duration_min": 45, "earliest_min": 11 * 60, "latest_min": 15 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R9", "title": "Recruiting", "duration_min": 30, "earliest_min": 11 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": False},
                ],
            },
            {
                "name": "reschedule_required",
                "seed_events": [
                    {"title": "Optional Sync", "start_min": 11 * 60, "end_min": 12 * 60, "locked": False, "priority": 1},
                    {"title": "Partner Call", "start_min": 13 * 60, "end_min": 14 * 60, "locked": False, "priority": 2},
                ],
                "requests": [
                    {"request_id": "R10", "title": "Board Prep", "duration_min": 90, "earliest_min": 11 * 60, "latest_min": 16 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R11", "title": "QA", "duration_min": 30, "earliest_min": 10 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": True},
                ],
            },
            {
                "name": "staggered_priorities",
                "seed_events": [
                    {"title": "Lunch", "start_min": 12 * 60, "end_min": 13 * 60, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R12", "title": "Enterprise Renewal", "duration_min": 60, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R13", "title": "Bug Triage", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 3, "must_schedule": True},
                    {"request_id": "R14", "title": "Culture Chat", "duration_min": 45, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 1, "must_schedule": False},
                ],
            },
            {
                "name": "zero_buffer_chain",
                "seed_events": [
                    {"title": "A", "start_min": 9 * 60, "end_min": 10 * 60, "locked": True, "priority": 1},
                    {"title": "B", "start_min": 10 * 60, "end_min": 11 * 60, "locked": True, "priority": 1},
                    {"title": "C", "start_min": 11 * 60, "end_min": 12 * 60, "locked": True, "priority": 1},
                    {"title": "D", "start_min": 13 * 60, "end_min": 14 * 60, "locked": True, "priority": 1},
                    {"title": "E", "start_min": 14 * 60, "end_min": 15 * 60, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R15", "title": "Escalation", "duration_min": 60, "earliest_min": 12 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R16", "title": "Debrief", "duration_min": 30, "earliest_min": 15 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": True},
                ],
            },
            {
                "name": "cross_noon_pressure",
                "seed_events": [
                    {"title": "Morning Ops", "start_min": 9 * 60 + 30, "end_min": 11 * 60, "locked": True, "priority": 2},
                    {"title": "Lunch", "start_min": 12 * 60, "end_min": 13 * 60, "locked": True, "priority": 1},
                    {"title": "Townhall", "start_min": 16 * 60, "end_min": 17 * 60, "locked": True, "priority": 3},
                ],
                "requests": [
                    {"request_id": "R17", "title": "Customer Escalation", "duration_min": 90, "earliest_min": 11 * 60, "latest_min": 16 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R18", "title": "Interview", "duration_min": 45, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 3, "must_schedule": True},
                ],
            },
            {
                "name": "locked_wall",
                "seed_events": [
                    {"title": "Block1", "start_min": 9 * 60, "end_min": 10 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Block2", "start_min": 10 * 60 + 45, "end_min": 12 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Block3", "start_min": 13 * 60, "end_min": 14 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Block4", "start_min": 15 * 60, "end_min": 17 * 60, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R19", "title": "War Room", "duration_min": 60, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R20", "title": "Retro", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": False},
                ],
            },
            {
                "name": "long_meeting_tradeoff",
                "seed_events": [
                    {"title": "Platform Sync", "start_min": 10 * 60, "end_min": 11 * 60, "locked": False, "priority": 1},
                    {"title": "Partner Review", "start_min": 13 * 60, "end_min": 14 * 60, "locked": False, "priority": 2},
                ],
                "requests": [
                    {"request_id": "R21", "title": "Quarterly Planning", "duration_min": 120, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R22", "title": "Legal Review", "duration_min": 45, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 3, "must_schedule": True},
                ],
            },
            {
                "name": "many_optionals_noise",
                "seed_events": [
                    {"title": "Core Meeting", "start_min": 11 * 60, "end_min": 12 * 60, "locked": True, "priority": 2},
                ],
                "requests": [
                    {"request_id": "R23", "title": "Incident Deep Dive", "duration_min": 60, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R24", "title": "Mentoring", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 1, "must_schedule": False},
                    {"request_id": "R25", "title": "Brainstorm", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 1, "must_schedule": False},
                    {"request_id": "R26", "title": "Follow-up", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 1, "must_schedule": False},
                ],
            },
            {
                "name": "late_day_deadline",
                "seed_events": [
                    {"title": "Morning Focus", "start_min": 9 * 60, "end_min": 11 * 60, "locked": True, "priority": 1},
                    {"title": "Afternoon Focus", "start_min": 13 * 60, "end_min": 15 * 60, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R27", "title": "End-of-day Escalation", "duration_min": 45, "earliest_min": 16 * 60, "latest_min": 17 * 60, "priority": 5, "must_schedule": True},
                    {"request_id": "R28", "title": "Status Review", "duration_min": 30, "earliest_min": 15 * 60, "latest_min": 17 * 60, "priority": 3, "must_schedule": True},
                ],
            },
            {
                "name": "micro_slot_fragmentation",
                "seed_events": [
                    {"title": "Slot1", "start_min": 9 * 60, "end_min": 9 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Slot2", "start_min": 10 * 60, "end_min": 10 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Slot3", "start_min": 11 * 60, "end_min": 11 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Slot4", "start_min": 13 * 60, "end_min": 13 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Slot5", "start_min": 14 * 60, "end_min": 14 * 60 + 30, "locked": True, "priority": 1},
                    {"title": "Slot6", "start_min": 15 * 60, "end_min": 15 * 60 + 30, "locked": True, "priority": 1},
                ],
                "requests": [
                    {"request_id": "R29", "title": "Needs 60 mins", "duration_min": 60, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 4, "must_schedule": True},
                    {"request_id": "R30", "title": "Needs 30 mins", "duration_min": 30, "earliest_min": 9 * 60, "latest_min": 17 * 60, "priority": 2, "must_schedule": True},
                ],
            },
        ]

    def _base_day(self) -> datetime:
        now = datetime.now(tz=self.tz)
        base = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return base + timedelta(days=1)

    def _minutes_to_dt(self, day: datetime, minute_of_day: int) -> datetime:
        return day + timedelta(minutes=int(minute_of_day))

    def _event_conflicts(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return a["start"] < b["end"] and b["start"] < a["end"]

    def _find_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        for ev in self.events:
            if ev["id"] == event_id:
                return ev
        return None

    def _working_hours(self) -> Tuple[datetime, datetime]:
        start = self.day_anchor.replace(hour=self.work_start_hour, minute=0)
        end = self.day_anchor.replace(hour=self.work_end_hour, minute=0)
        return start, end

    def _is_inside_work_hours(self, start: datetime, end: datetime) -> bool:
        ws, we = self._working_hours()
        return ws <= start < end <= we

    def _has_conflict(self, candidate: Dict[str, Any], ignore_event_id: Optional[str] = None) -> bool:
        for ev in self.events:
            if ignore_event_id and ev["id"] == ignore_event_id:
                continue
            if self._event_conflicts(candidate, ev):
                return True
        return False

    def _request_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        for req in self.requests:
            if req["request_id"] == request_id:
                return req
        return None

    def _pending_requests(self) -> List[Dict[str, Any]]:
        return [r for r in self.requests if r["request_id"] not in self.completed_requests]

    def _build_observation(self) -> str:
        pending = self._pending_requests()
        payload = {
            "backend": self.backend,
            "scenario": self.scenario_name,
            "episode_id": self.episode_id,
            "step": self.current_step,
            "remaining_steps": max(self.max_steps - self.current_step, 0),
            "time_generated": _now_utc_iso(),
            "events": [
                {
                    "id": ev["id"],
                    "title": ev["title"],
                    "start": _to_iso(ev["start"]),
                    "end": _to_iso(ev["end"]),
                    "locked": ev["locked"],
                    "priority": ev["priority"],
                    "source": ev["source"],
                    "request_id": ev.get("request_id"),
                }
                for ev in sorted(self.events, key=lambda x: x["start"])
            ],
            "pending_requests": pending,
            "completed_requests": self.completed_requests,
            "last_error": self.last_error,
            "last_action": self.last_action,
        }
        return json.dumps(payload, separators=(",", ":"))

    def _insert_live_event(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        body = {
            "summary": f"{self.managed_prefix} {ev['title']}",
            "description": json.dumps(
                {
                    "rlgym": True,
                    "episode_id": self.episode_id,
                    "locked": ev["locked"],
                    "priority": ev["priority"],
                    "source": ev["source"],
                    "request_id": ev.get("request_id"),
                }
            ),
            "start": {"dateTime": _to_iso(ev["start"])},
            "end": {"dateTime": _to_iso(ev["end"])},
        }
        created = self._execute_with_retries(
            lambda: self._service.events()
            .insert(calendarId=self.calendar_id, body=body, sendUpdates="none")
            .execute(num_retries=0),
            "insert_event",
        )
        ev = dict(ev)
        ev["id"] = created["id"]
        return ev

    def _patch_live_event_time(self, event_id: str, start: datetime, end: datetime) -> None:
        self._execute_with_retries(
            lambda: self._service.events()
            .patch(
                calendarId=self.calendar_id,
                eventId=event_id,
                body={"start": {"dateTime": _to_iso(start)}, "end": {"dateTime": _to_iso(end)}},
                sendUpdates="none",
            )
            .execute(num_retries=0),
            "patch_event_time",
        )

    def _delete_live_event(self, event_id: str) -> None:
        self._execute_with_retries(
            lambda: self._service.events()
            .delete(calendarId=self.calendar_id, eventId=event_id, sendUpdates="none")
            .execute(num_retries=0),
            "delete_event",
            ignore_http_statuses={404},
        )

    def _clear_live_managed_events(self) -> None:
        day_start = self.day_anchor.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=2)
        token = None
        while True:
            resp = self._execute_with_retries(
                lambda: self._service.events()
                .list(
                    calendarId=self.calendar_id,
                    timeMin=_to_iso(day_start),
                    timeMax=_to_iso(day_end),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                    pageToken=token,
                )
                .execute(num_retries=0),
                "list_events_for_clear",
            )

            for item in resp.get("items", []):
                summary = item.get("summary", "")
                if summary.startswith(self.managed_prefix):
                    self._delete_live_event(item["id"])

            token = resp.get("nextPageToken")
            if not token:
                break

    def _seed_episode(self, forced_scenario_name: Optional[str] = None) -> None:
        templates = self._scenario_templates()
        template = None
        if forced_scenario_name:
            for item in templates:
                if item["name"] == forced_scenario_name:
                    template = item
                    break
            if template is None:
                raise ValueError(f"Unknown scenario_name: {forced_scenario_name}")
        else:
            template = self.np_random.choice(templates)
        self.scenario_name = str(template["name"])
        self.events = []
        self.requests = [dict(r) for r in template["requests"]]
        self.completed_requests = {}
        self._rewarded_requests = {}

        for raw in template["seed_events"]:
            start = self._minutes_to_dt(self.day_anchor, int(raw["start_min"]))
            end = self._minutes_to_dt(self.day_anchor, int(raw["end_min"]))
            ev = {
                "id": f"sim_{uuid.uuid4().hex[:10]}",
                "title": str(raw["title"]),
                "start": start,
                "end": end,
                "locked": bool(raw["locked"]),
                "priority": int(raw["priority"]),
                "source": "seed",
            }
            self.events.append(ev)

        if self.backend == "live":
            self._clear_live_managed_events()
            seeded_live = []
            for ev in self.events:
                seeded_live.append(self._insert_live_event(ev))
            self.events = seeded_live

    def _validate_request_window(self, req: Dict[str, Any], start: datetime, end: datetime) -> bool:
        earliest = self._minutes_to_dt(self.day_anchor, int(req["earliest_min"]))
        latest = self._minutes_to_dt(self.day_anchor, int(req["latest_min"]))
        return earliest <= start and end <= latest

    def _auto_slot_for_request(self, req: Dict[str, Any]) -> Optional[Tuple[datetime, datetime]]:
        duration = int(req["duration_min"])
        earliest = self._minutes_to_dt(self.day_anchor, int(req["earliest_min"]))
        latest = self._minutes_to_dt(self.day_anchor, int(req["latest_min"]))

        cursor = earliest
        while cursor + timedelta(minutes=duration) <= latest:
            end = cursor + timedelta(minutes=duration)
            candidate = {
                "id": "candidate",
                "title": req["title"],
                "start": cursor,
                "end": end,
                "locked": False,
                "priority": int(req["priority"]),
                "source": "scheduled",
                "request_id": req["request_id"],
            }
            if self._is_inside_work_hours(cursor, end) and not self._has_conflict(candidate):
                return cursor, end
            cursor += timedelta(minutes=15)

        return None

    def _schedule_request(self, request_id: str, start_iso: str) -> Tuple[bool, str, Optional[str]]:
        req = self._request_by_id(request_id)
        if req is None:
            return False, f"Unknown request_id: {request_id}", None
        if request_id in self.completed_requests:
            return False, f"Request already scheduled: {request_id}", None

        start = _parse_iso(start_iso).astimezone(self.tz)
        end = start + timedelta(minutes=int(req["duration_min"]))

        candidate = {
            "id": f"sim_{uuid.uuid4().hex[:10]}",
            "title": req["title"],
            "start": start,
            "end": end,
            "locked": False,
            "priority": int(req["priority"]),
            "source": "scheduled",
            "request_id": request_id,
        }

        if not self._validate_request_window(req, start, end):
            return False, "Requested slot outside request window", None
        if not self._is_inside_work_hours(start, end):
            return False, "Requested slot outside work hours", None
        if self._has_conflict(candidate):
            return False, "Requested slot conflicts with existing events", None

        if self.backend == "live":
            candidate = self._insert_live_event(candidate)

        self.events.append(candidate)
        self.completed_requests[request_id] = candidate["id"]
        return True, "Scheduled request", request_id

    def _auto_schedule(self, request_id: str) -> Tuple[bool, str, Optional[str]]:
        req = self._request_by_id(request_id)
        if req is None:
            return False, f"Unknown request_id: {request_id}", None
        if request_id in self.completed_requests:
            return False, f"Request already scheduled: {request_id}", None

        slot = self._auto_slot_for_request(req)
        if slot is None:
            return False, "No valid slot found", None

        start, _ = slot
        return self._schedule_request(request_id, _to_iso(start))

    def _move_event(self, event_id: str, start_iso: str, end_iso: str) -> Tuple[bool, str, None]:
        ev = self._find_event(event_id)
        if ev is None:
            return False, f"Unknown event_id: {event_id}", None
        if ev["locked"]:
            return False, f"Event {event_id} is locked and cannot be moved", None

        new_start = _parse_iso(start_iso).astimezone(self.tz)
        new_end = _parse_iso(end_iso).astimezone(self.tz)
        if not (new_start < new_end):
            return False, "start_iso must be before end_iso", None
        if not self._is_inside_work_hours(new_start, new_end):
            return False, "Moved event must remain inside work hours", None

        candidate = dict(ev)
        candidate["start"] = new_start
        candidate["end"] = new_end
        if self._has_conflict(candidate, ignore_event_id=event_id):
            return False, "Moved event would conflict", None

        if self.backend == "live":
            self._patch_live_event_time(event_id=event_id, start=new_start, end=new_end)

        ev["start"] = new_start
        ev["end"] = new_end
        return True, "Moved event", None

    def _cancel_event(self, event_id: str) -> Tuple[bool, str, None]:
        ev = self._find_event(event_id)
        if ev is None:
            return False, f"Unknown event_id: {event_id}", None
        if ev["locked"]:
            return False, "Locked event cannot be cancelled", None

        if self.backend == "live":
            self._delete_live_event(event_id)

        self.events = [x for x in self.events if x["id"] != event_id]

        # If this event satisfied a request, put request back to pending.
        if ev.get("request_id"):
            self.completed_requests.pop(ev["request_id"], None)

        return True, "Cancelled event", None

    def _bulk_reschedule(self, payload_json: str) -> Tuple[bool, str, None]:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as exc:
            return False, f"Invalid payload_json: {exc}", None

        operations = payload.get("operations") if isinstance(payload, dict) else None
        if not isinstance(operations, list):
            return False, "payload_json must contain operations list", None

        for op in operations:
            if not isinstance(op, dict):
                return False, "Each operation must be an object", None

            op_type = op.get("type", "")
            if op_type == "move_event":
                ok, msg, _ = self._move_event(
                    event_id=str(op.get("event_id", "")),
                    start_iso=str(op.get("start_iso", "")),
                    end_iso=str(op.get("end_iso", "")),
                )
            elif op_type == "cancel_event":
                ok, msg, _ = self._cancel_event(event_id=str(op.get("event_id", "")))
            elif op_type == "auto_schedule":
                ok, msg, _ = self._auto_schedule(request_id=str(op.get("request_id", "")))
            else:
                return False, f"Unsupported bulk op: {op_type}", None

            if not ok:
                return False, f"Bulk op failed: {msg}", None

        return True, "Bulk reschedule completed", None

    def _apply_action(self, action: Dict[str, str]) -> Tuple[bool, str, Optional[str]]:
        action_type = action.get("type", "").strip()

        if action_type == "schedule_request":
            request_id = action.get("request_id", "").strip()
            start_iso = action.get("start_iso", "").strip()
            if not request_id or not start_iso:
                return False, "schedule_request requires request_id and start_iso", None
            return self._schedule_request(request_id, start_iso)

        if action_type == "auto_schedule":
            request_id = action.get("request_id", "").strip()
            if not request_id:
                return False, "auto_schedule requires request_id", None
            return self._auto_schedule(request_id)

        if action_type == "move_event":
            event_id = action.get("event_id", "").strip()
            start_iso = action.get("start_iso", "").strip()
            end_iso = action.get("end_iso", "").strip()
            if not event_id or not start_iso or not end_iso:
                return False, "move_event requires event_id/start_iso/end_iso", None
            return self._move_event(event_id, start_iso, end_iso)

        if action_type == "cancel_event":
            event_id = action.get("event_id", "").strip()
            if not event_id:
                return False, "cancel_event requires event_id", None
            return self._cancel_event(event_id)

        if action_type == "bulk_reschedule":
            return self._bulk_reschedule(action.get("payload_json", ""))

        if action_type == "noop":
            return True, "No operation", None

        return False, f"Unsupported action type: {action_type}", None

    def _compute_goal_metrics(self) -> Dict[str, Any]:
        pending = self._pending_requests()
        high_priority_pending = [r for r in pending if int(r["priority"]) >= 4 and bool(r["must_schedule"])]

        conflict_count = 0
        for i in range(len(self.events)):
            for j in range(i + 1, len(self.events)):
                if self._event_conflicts(self.events[i], self.events[j]):
                    conflict_count += 1

        outside_hours = 0
        for ev in self.events:
            if not self._is_inside_work_hours(ev["start"], ev["end"]):
                outside_hours += 1

        all_required_scheduled = True
        for req in self.requests:
            if bool(req["must_schedule"]) and req["request_id"] not in self.completed_requests:
                all_required_scheduled = False
                break

        return {
            "pending_requests": len(pending),
            "high_priority_pending": len(high_priority_pending),
            "conflict_count": conflict_count,
            "outside_hours_events": outside_hours,
            "all_required_scheduled": all_required_scheduled,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.last_action = {}
        self.last_error = None
        self.episode_id = uuid.uuid4().hex[:12]
        self.day_anchor = self._base_day()

        # Pacing helps avoid API burst issues in live mode when many resets happen.
        if self.backend == "live":
            time.sleep(self.read_backoff_seconds)

        forced_name = None
        if options and isinstance(options, dict):
            forced_name = options.get("scenario_name")
        self._seed_episode(forced_scenario_name=forced_name)
        obs = self._build_observation()
        info = {
            "scenario": self.scenario_name,
            "backend": self.backend,
            "goal_metrics": self._compute_goal_metrics(),
            "action_help": [
                "schedule_request",
                "auto_schedule",
                "move_event",
                "cancel_event",
                "bulk_reschedule",
                "noop",
            ],
        }
        return obs, info

    def step(self, action: Dict[str, str]):
        self.current_step += 1
        self.last_action = dict(action)
        self.last_error = None

        reward = -1.0  # step penalty
        invalid_call = False

        try:
            ok, msg, newly_completed_request = self._apply_action(action)
            if not ok:
                invalid_call = True
                self.last_error = msg
                reward -= 6.0
            else:
                # Reward successful scheduling action on first completion.
                if newly_completed_request and not self._rewarded_requests.get(newly_completed_request, False):
                    req = self._request_by_id(newly_completed_request)
                    priority = int(req["priority"]) if req else 1
                    reward += float(priority) * 2.0
                    self._rewarded_requests[newly_completed_request] = True
        except RuntimeError as exc:
            invalid_call = True
            self.last_error = str(exc)
            reward -= 6.0

        metrics = self._compute_goal_metrics()

        # Global penalties/bonuses to push safer schedules.
        reward -= float(metrics["conflict_count"]) * 3.0
        reward -= float(metrics["outside_hours_events"]) * 2.0

        terminated = False
        if metrics["all_required_scheduled"] and metrics["conflict_count"] == 0 and metrics["outside_hours_events"] == 0:
            reward += 10.0
            terminated = True

        truncated = self.current_step >= self.max_steps and not terminated
        observation = self._build_observation()

        info = {
            "backend": self.backend,
            "scenario": self.scenario_name,
            "step": self.current_step,
            "valid_call": not invalid_call,
            "error": self.last_error,
            "goal_metrics": metrics,
        }
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    raise SystemExit("Import CalendarSchedulingEnv from calendar_env.py")
