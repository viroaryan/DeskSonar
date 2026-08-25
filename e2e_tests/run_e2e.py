"""
DeskSonar Standalone Automated E2E Test Runner (Milestone 5)
Executes Tiers 1-4 with comprehensive reporting, tier statistics, and return codes.
"""
import os
import sys
import time
from pathlib import Path
import pytest


def print_banner():
    banner = """
================================================================================
           DeskSonar Acoustic Ultrasonic Radar & Air Mouse System
                   Requirement-Driven Opaque-Box E2E Test Suite
================================================================================
Tiers:
  [Tier 1] Feature Coverage (F1 to F10 Happy Paths)
  [Tier 2] Boundary & Corner Cases (Kinematic Extrema & Geofence Margins)
  [Tier 3] Cross-Feature Combinations & Concurrency Matrix
  [Tier 4] Real-World Application Scenarios & Workloads
================================================================================
"""
    print(banner)


def run_e2e_suite():
    print_banner()
    tests_dir = Path(__file__).resolve().parent
    root_dir = tests_dir.parent

    # Ensure DeskSonar root is in sys.path
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    tiers = [
        ("Tier 1 — Feature Coverage (F1-F10)", str(tests_dir / "test_tier1_features.py")),
        ("Tier 2 — Boundary & Corner Cases (F1-F10)", str(tests_dir / "test_tier2_boundaries.py")),
        ("Tier 3 — Cross-Feature Combinations", str(tests_dir / "test_tier3_combinations.py")),
        ("Tier 4 — Real-World Application Scenarios", str(tests_dir / "test_tier4_scenarios.py")),
    ]

    total_start = time.perf_counter()
    tier_results = []
    all_passed = True

    for tier_name, tier_path in tiers:
        print(f"\n>> Executing {tier_name}...")
        t_start = time.perf_counter()

        ret = pytest.main([
            tier_path,
            "-v",
            "--tb=short",
            "-o", "junit_family=xunit2"
        ])
        t_elapsed = time.perf_counter() - t_start
        passed = (ret == pytest.ExitCode.OK)
        tier_results.append((tier_name, passed, t_elapsed, ret))
        if not passed:
            all_passed = False

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 80)
    print("                           E2E TEST SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Tier':<50} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 80)

    for tier_name, passed, t_elapsed, _ in tier_results:
        status_str = "PASSED" if passed else "FAILED"
        print(f"{tier_name:<50} | {status_str:<10} | {t_elapsed:.2f}s")

    print("-" * 80)
    total_status = "ALL TIERS PASSED" if all_passed else "SOME TIERS FAILED"
    print(f"Overall Result: {total_status} (Total Time: {total_elapsed:.2f}s)")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = run_e2e_suite()
    sys.exit(exit_code)
