import argparse
import json
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from calendar_env import CalendarSchedulingEnv


Action = Dict[str, str]
PolicyFn = Callable[[str, int], Tuple[Action, str]]


def blank_action() -> Action:
    return {
        "type": "",
        "event_id": "",
        "request_id": "",
        "start_iso": "",
        "end_iso": "",
        "title": "",
        "payload_json": "",
    }


def act(**kwargs: str) -> Action:
    data = blank_action()
    data.update(kwargs)
    return data


def parse_obs(obs: str) -> Dict:
    return json.loads(obs)


def _request_slot_start(req: Dict) -> str:
    # Heuristic deterministic slot: place at earliest window.
    day = req.get("day_anchor")
    if day:
        pass
    # We only need this in tests using auto_schedule mostly.
    return ""


def robust_policy(obs: str, step_idx: int) -> Tuple[Action, str]:
    payload = parse_obs(obs)
    pending = payload["pending_requests"]

    if not pending:
        return act(type="noop"), "done"

    # Prioritize highest priority and must_schedule first.
    pending_sorted = sorted(
        pending,
        key=lambda r: (int(r["must_schedule"]) * 10 + int(r["priority"]), int(r["duration_min"])),
        reverse=True,
    )
    target = pending_sorted[0]

    # Use auto schedule to keep policy robust across scenarios.
    return act(type="auto_schedule", request_id=str(target["request_id"])), "auto schedule next request"


def invalid_then_recover_policy(obs: str, step_idx: int) -> Tuple[Action, str]:
    if step_idx == 0:
        return act(type="unsupported_action"), "invalid action type"
    if step_idx == 1:
        return act(type="move_event", event_id="missing", start_iso="2026-01-01T10:00:00-05:00", end_iso="2026-01-01T10:30:00-05:00"), "invalid event id"
    return robust_policy(obs, step_idx)


def truncation_policy(obs: str, step_idx: int) -> Tuple[Action, str]:
    # Waste steps deliberately.
    return act(type="noop"), "intentional no-op"


def conflict_seeking_policy(obs: str, step_idx: int) -> Tuple[Action, str]:
    payload = parse_obs(obs)
    pending = payload["pending_requests"]
    events = payload["events"]
    if not pending or not events:
        return robust_policy(obs, step_idx)

    # Try to move an unlocked event into conflict first, then recover.
    if step_idx == 0:
        moving = next((e for e in events if not e["locked"]), None)
        anchor = events[0]
        if moving is None:
            moving = anchor
        return (
            act(
                type="move_event",
                event_id=str(moving["id"]),
                start_iso=str(anchor["start"]),
                end_iso=str(anchor["end"]),
            ),
            "force conflict/locked failure",
        )
    return robust_policy(obs, step_idx)


def random_fuzz_policy_factory(seed: int) -> PolicyFn:
    rng = random.Random(seed)

    def policy(obs: str, step_idx: int) -> Tuple[Action, str]:
        payload = parse_obs(obs)
        pending = payload.get("pending_requests", [])
        events = payload.get("events", [])

        draw = rng.random()
        if draw < 0.15:
            return act(type="unsupported"), "random invalid action"
        if draw < 0.30:
            return act(type="bulk_reschedule", payload_json='{"operations":"bad"}'), "random bad payload"
        if pending and draw < 0.85:
            req = rng.choice(pending)
            return act(type="auto_schedule", request_id=str(req["request_id"])), "random auto schedule"
        if events:
            ev = rng.choice(events)
            return act(type="cancel_event", event_id=str(ev["id"])), "random cancel"
        return act(type="noop"), "random noop"

    return policy


@dataclass
class CaseResult:
    name: str
    terminated: bool
    truncated: bool
    total_reward: float
    invalid_calls: int
    steps: int
    scenario: str


def run_case(env: CalendarSchedulingEnv, case_name: str, policy: PolicyFn, seed: int, max_steps: int) -> CaseResult:
    env.max_steps = max_steps
    obs, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    total_reward = 0.0
    invalid_calls = 0
    step_idx = 0

    print(f"\n=== CASE {case_name} | scenario={info['scenario']} | seed={seed} ===")

    while not (terminated or truncated):
        action, rationale = policy(obs, step_idx)
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        if not step_info.get("valid_call", True):
            invalid_calls += 1

        print(
            f"step={step_idx:02d} action={action['type']} rationale={rationale} "
            f"reward={reward:.1f} term={terminated} trunc={truncated} "
            f"pending={step_info['goal_metrics']['pending_requests']} invalid={not step_info['valid_call']}"
        )
        step_idx += 1

    print(
        f"result: reward={total_reward:.1f} steps={step_idx} invalid_calls={invalid_calls} "
        f"terminated={terminated} truncated={truncated}"
    )

    return CaseResult(
        name=case_name,
        terminated=terminated,
        truncated=truncated,
        total_reward=total_reward,
        invalid_calls=invalid_calls,
        steps=step_idx,
        scenario=info["scenario"],
    )


def assert_case(result: CaseResult, expectation: str) -> None:
    if expectation == "should_terminate":
        assert result.terminated, f"{result.name} expected terminated=True"
    elif expectation == "should_truncate":
        assert result.truncated, f"{result.name} expected truncated=True"
    elif expectation == "should_have_invalid":
        assert result.invalid_calls >= 1, f"{result.name} expected invalid calls"
    elif expectation == "no_assert":
        return
    else:
        raise ValueError(f"Unknown expectation: {expectation}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress tests for CalendarSchedulingEnv")
    parser.add_argument("--backend", choices=["simulated", "live"], default="simulated")
    parser.add_argument("--calendar-id", default="")
    parser.add_argument("--credentials-path", default="")
    args = parser.parse_args()

    env = CalendarSchedulingEnv(
        backend=args.backend,
        calendar_id=args.calendar_id or None,
        credentials_path=args.credentials_path or None,
        max_steps=30,
    )

    cases = [
        ("01_happy_seed2", robust_policy, 2, 20, "should_terminate"),
        ("02_happy_seed7", robust_policy, 7, 20, "should_terminate"),
        ("03_happy_seed11", robust_policy, 11, 20, "should_terminate"),
        ("04_invalid_then_recover", invalid_then_recover_policy, 5, 25, "should_have_invalid"),
        ("05_truncation_guard", truncation_policy, 3, 4, "should_truncate"),
        ("06_conflict_then_recover", conflict_seeking_policy, 9, 25, "should_have_invalid"),
        ("07_fuzz_seed1", random_fuzz_policy_factory(1), 4, 20, "no_assert"),
        ("08_fuzz_seed2", random_fuzz_policy_factory(2), 6, 20, "no_assert"),
        ("09_fuzz_seed3", random_fuzz_policy_factory(3), 8, 20, "no_assert"),
        ("10_fuzz_seed4", random_fuzz_policy_factory(4), 10, 20, "no_assert"),
        ("11_priority_pressure", robust_policy, 13, 18, "should_terminate"),
        ("12_soak_short", robust_policy, 15, 18, "should_terminate"),
    ]

    results: List[CaseResult] = []
    for name, policy, seed, max_steps, expectation in cases:
        result = run_case(env, name, policy, seed, max_steps)
        assert_case(result, expectation)
        results.append(result)

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r.name}: scenario={r.scenario} steps={r.steps} reward={r.total_reward:.1f} "
            f"invalid={r.invalid_calls} term={r.terminated} trunc={r.truncated}"
        )

    print("\nAll calendar stress tests finished.")


if __name__ == "__main__":
    main()
