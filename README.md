# Google Calendar Scheduling RL Gym

A Gymnasium environment (`CalendarSchedulingEnv`) for calendar scheduling with both simulated and live Google Calendar backends.

## Features
- RL-style `reset()` / `step(action)` interface
- Simulated backend for aggressive stress testing
- Live backend for real Google Calendar API execution
- Actions: `schedule_request`, `auto_schedule`, `move_event`, `cancel_event`, `bulk_reschedule`, `noop`
- Reward shaping for scheduling quality (priority, conflict avoidance, work-hour compliance)
- 12 stress tests including invalid-action recovery, truncation, and fuzz cases

## Setup
```powershell
pip install -r requirements.txt
```

## Run stress tests (simulated)
```powershell
python calendar_stress_tests.py --backend simulated
```

## Run stress tests (live backend)
```powershell
python calendar_stress_tests.py --backend live --calendar-id "<CALENDAR_ID>" --credentials-path "C:\path\to\service-account.json"
```

## Notes
- Enable Google Calendar API in your GCP project.
- Share your calendar with the service-account email if using live mode.
- Keep credentials private.
