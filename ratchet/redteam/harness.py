"""
RATCHET Red Team Harness

Orchestrates execution of adversarial attacks (RT-01 through RT-05) against
RATCHET components. Provides structured testing, reporting, and integration
with the detection and federation engines.

Usage:
    harness = RedTeamHarness()
    suite = AttackSuite.full_suite()
    results = harness.run_suite(suite, target)
    report = harness.generate_report(results)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .attacks import (
    Attack,
    AttackResult,
    AttackSeverity,
    RT01_EmergentDeception,
    RT02_SlowCapture,
    RT03_Mimicry,
    RT04_Flooding,
    RT05_DiverseSybils,
)


class SuiteType(Enum):
    """Types of attack suites."""
    MINIMAL = "minimal"      # Quick smoke test (RT-04 only)
    STANDARD = "standard"    # Common attacks (RT-02, RT-03, RT-04)
    FULL = "full"            # All attacks (RT-01 through RT-05)
    CRITICAL = "critical"    # Critical severity only (RT-01, RT-02, RT-05)


@dataclass
class AttackConfiguration:
    """Configuration for a single attack instance."""
    attack_class: Type[Attack]
    params: Dict[str, Any] = field(default_factory=dict)
    iterations: int = 100
    enabled: bool = True
    run_kwargs: Dict[str, Any] = field(default_factory=dict)

    def create_attack(self) -> Attack:
        """Instantiate the configured attack."""
        return self.attack_class(**self.params)


@dataclass
class SuiteResult:
    """Aggregate result of running an attack suite."""
    suite_name: str
    start_time: float
    end_time: float
    attack_results: List[AttackResult]
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_attacks(self) -> int:
        return len(self.attack_results)

    @property
    def successful_attacks(self) -> int:
        return sum(1 for r in self.attack_results if r.success)

    @property
    def blocked_attacks(self) -> int:
        return self.total_attacks - self.successful_attacks

    @property
    def critical_successes(self) -> int:
        return sum(
            1 for r in self.attack_results
            if r.success and r.severity == AttackSeverity.CRITICAL
        )

    @property
    def overall_security_score(self) -> float:
        """
        Compute overall security score (0-100).

        Weighting:
        - Critical attacks: 40 points each
        - High attacks: 20 points each
        - Medium attacks: 10 points each
        - Low attacks: 5 points each
        """
        if not self.attack_results:
            return 100.0

        severity_weights = {
            AttackSeverity.CRITICAL: 40,
            AttackSeverity.HIGH: 20,
            AttackSeverity.MEDIUM: 10,
            AttackSeverity.LOW: 5,
        }

        max_score = sum(severity_weights[r.severity] for r in self.attack_results)
        lost_points = sum(
            severity_weights[r.severity] * r.success_rate
            for r in self.attack_results
        )

        if max_score == 0:
            return 100.0

        return max(0, 100 * (1 - lost_points / max_score))


class AttackSuite:
    """
    Collection of attacks to run together.

    Provides pre-configured suites and custom suite building.
    """

    def __init__(
        self,
        name: str,
        attacks: Optional[List[AttackConfiguration]] = None,
    ):
        """
        Initialize attack suite.

        Args:
            name: Suite identifier
            attacks: List of attack configurations
        """
        self.name = name
        self.attacks = attacks or []

    def add_attack(
        self,
        attack_class: Type[Attack],
        params: Optional[Dict[str, Any]] = None,
        iterations: int = 100,
        **run_kwargs
    ) -> 'AttackSuite':
        """
        Add an attack to the suite.

        Args:
            attack_class: The Attack class to add
            params: Parameters for attack initialization
            iterations: Number of iterations to run
            **run_kwargs: Additional kwargs for execute()

        Returns:
            Self for chaining
        """
        config = AttackConfiguration(
            attack_class=attack_class,
            params=params or {},
            iterations=iterations,
            run_kwargs=run_kwargs,
        )
        self.attacks.append(config)
        return self

    def enable_all(self) -> 'AttackSuite':
        """Enable all attacks in the suite."""
        for config in self.attacks:
            config.enabled = True
        return self

    def disable_attack(self, attack_id: str) -> 'AttackSuite':
        """Disable a specific attack by ID."""
        for config in self.attacks:
            if config.attack_class.attack_id == attack_id:
                config.enabled = False
        return self

    @classmethod
    def minimal(cls, iterations: int = 50) -> 'AttackSuite':
        """
        Create minimal suite for quick smoke testing.

        Includes only RT-04 (Flooding) as it's fastest to run.
        """
        suite = cls(name="Minimal")
        suite.add_attack(RT04_Flooding, iterations=iterations)
        return suite

    @classmethod
    def standard(cls, iterations: int = 100) -> 'AttackSuite':
        """
        Create standard suite with common attacks.

        Includes RT-02, RT-03, RT-04.
        """
        suite = cls(name="Standard")
        suite.add_attack(RT02_SlowCapture, iterations=iterations)
        suite.add_attack(RT03_Mimicry, iterations=iterations)
        suite.add_attack(RT04_Flooding, iterations=iterations)
        return suite

    @classmethod
    def full_suite(cls, iterations: int = 100) -> 'AttackSuite':
        """
        Create full suite with all attacks.

        Includes RT-01 through RT-05.
        """
        suite = cls(name="Full")
        suite.add_attack(RT01_EmergentDeception, iterations=iterations)
        suite.add_attack(RT02_SlowCapture, iterations=iterations)
        suite.add_attack(RT03_Mimicry, iterations=iterations)
        suite.add_attack(RT04_Flooding, iterations=iterations)
        suite.add_attack(RT05_DiverseSybils, iterations=iterations)
        return suite

    @classmethod
    def critical_only(cls, iterations: int = 100) -> 'AttackSuite':
        """
        Create suite with only critical severity attacks.

        Includes RT-01, RT-02, RT-05.
        """
        suite = cls(name="Critical")
        suite.add_attack(RT01_EmergentDeception, iterations=iterations)
        suite.add_attack(RT02_SlowCapture, iterations=iterations)
        suite.add_attack(RT05_DiverseSybils, iterations=iterations)
        return suite

    @classmethod
    def from_type(cls, suite_type: SuiteType, iterations: int = 100) -> 'AttackSuite':
        """Create suite from SuiteType enum."""
        if suite_type == SuiteType.MINIMAL:
            return cls.minimal(iterations)
        elif suite_type == SuiteType.STANDARD:
            return cls.standard(iterations)
        elif suite_type == SuiteType.FULL:
            return cls.full_suite(iterations)
        elif suite_type == SuiteType.CRITICAL:
            return cls.critical_only(iterations)
        else:
            raise ValueError(f"Unknown suite type: {suite_type}")


class RedTeamHarness:
    """
    Orchestrates adversarial testing of RATCHET components.

    Provides:
    - Structured attack execution
    - Progress tracking
    - Result aggregation
    - Report generation

    Usage:
        harness = RedTeamHarness()

        # Run single attack
        result = harness.run_attack(RT01_EmergentDeception(), target)

        # Run full suite
        suite = AttackSuite.full_suite()
        results = harness.run_suite(suite, target)

        # Generate report
        report = harness.generate_report(results)
    """

    def __init__(
        self,
        verbose: bool = True,
        stop_on_critical: bool = False,
    ):
        """
        Initialize red team harness.

        Args:
            verbose: Whether to print progress
            stop_on_critical: Stop suite on critical attack success
        """
        self.verbose = verbose
        self.stop_on_critical = stop_on_critical
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(
        self,
        callback: Callable[[str, float], None]
    ) -> None:
        """
        Set callback for progress updates.

        Args:
            callback: Function taking (attack_id, progress_fraction)
        """
        self._progress_callback = callback

    def run_attack(
        self,
        attack: Attack,
        target: Any,
        iterations: int = 100,
        **kwargs
    ) -> AttackResult:
        """
        Run a single attack.

        Args:
            attack: The attack instance to execute
            target: The target component/system
            iterations: Number of iterations
            **kwargs: Additional parameters for execute()

        Returns:
            AttackResult with attack outcome
        """
        if self.verbose:
            print(f"Running {attack.attack_id}: {attack.attack_name}...")

        start = time.time()
        result = attack.execute(target, iterations=iterations, **kwargs)
        elapsed = time.time() - start

        if self.verbose:
            status = "SUCCEEDED" if result.success else "BLOCKED"
            print(f"  {status} in {elapsed:.2f}s")
            print(f"  Success rate: {result.success_rate:.1%}")
            print(f"  Detection rate: {result.detection_rate:.1%}")

        return result

    def run_suite(
        self,
        suite: AttackSuite,
        target: Any,
        detectors: Optional[Dict[str, Callable]] = None,
    ) -> SuiteResult:
        """
        Run a suite of attacks.

        Args:
            suite: The attack suite to run
            target: The target component/system
            detectors: Optional dict mapping attack_id to detector functions

        Returns:
            SuiteResult with aggregated results
        """
        detectors = detectors or {}
        results: List[AttackResult] = []

        start_time = time.time()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Running Attack Suite: {suite.name}")
            print(f"{'='*60}\n")

        enabled_attacks = [c for c in suite.attacks if c.enabled]
        total = len(enabled_attacks)

        for i, config in enumerate(enabled_attacks):
            # Create attack instance
            attack = config.create_attack()

            # Get detector if available
            detector = detectors.get(attack.attack_id)

            # Merge run kwargs
            kwargs = dict(config.run_kwargs)
            if detector:
                kwargs['detector'] = detector

            # Run attack
            result = self.run_attack(
                attack,
                target,
                iterations=config.iterations,
                **kwargs
            )
            results.append(result)

            # Update progress
            if self._progress_callback:
                self._progress_callback(attack.attack_id, (i + 1) / total)

            # Check for early termination
            if self.stop_on_critical:
                if result.success and result.severity == AttackSeverity.CRITICAL:
                    if self.verbose:
                        print(f"\nStopping: Critical attack {attack.attack_id} succeeded")
                    break

        end_time = time.time()

        # Build summary
        summary = self._build_summary(results)

        return SuiteResult(
            suite_name=suite.name,
            start_time=start_time,
            end_time=end_time,
            attack_results=results,
            summary=summary,
        )

    def _build_summary(self, results: List[AttackResult]) -> Dict[str, Any]:
        """Build summary statistics from results."""
        if not results:
            return {}

        by_severity = {}
        for sev in AttackSeverity:
            sev_results = [r for r in results if r.severity == sev]
            if sev_results:
                by_severity[sev.value] = {
                    'count': len(sev_results),
                    'successes': sum(1 for r in sev_results if r.success),
                    'avg_success_rate': sum(r.success_rate for r in sev_results) / len(sev_results),
                    'avg_detection_rate': sum(r.detection_rate for r in sev_results) / len(sev_results),
                }

        return {
            'total_attacks': len(results),
            'successful_attacks': sum(1 for r in results if r.success),
            'blocked_attacks': sum(1 for r in results if not r.success),
            'avg_success_rate': sum(r.success_rate for r in results) / len(results),
            'avg_detection_rate': sum(r.detection_rate for r in results) / len(results),
            'avg_evasion_rate': sum(r.evasion_rate for r in results) / len(results),
            'by_severity': by_severity,
        }

    def generate_report(
        self,
        suite_result: SuiteResult,
        format: str = 'text',
    ) -> str:
        """
        Generate a report from suite results.

        Args:
            suite_result: The SuiteResult to report on
            format: 'text' or 'json'

        Returns:
            Formatted report string
        """
        if format == 'json':
            return self._generate_json_report(suite_result)
        else:
            return self._generate_text_report(suite_result)

    def _generate_text_report(self, result: SuiteResult) -> str:
        """Generate human-readable text report."""
        lines = [
            "",
            "=" * 70,
            "RATCHET RED TEAM REPORT",
            "=" * 70,
            "",
            f"Suite: {result.suite_name}",
            f"Timestamp: {datetime.fromtimestamp(result.start_time).isoformat()}",
            f"Duration: {result.duration_seconds:.2f}s",
            "",
            "-" * 70,
            "SUMMARY",
            "-" * 70,
            "",
            f"Total Attacks:      {result.total_attacks}",
            f"Successful:         {result.successful_attacks}",
            f"Blocked:            {result.blocked_attacks}",
            f"Critical Successes: {result.critical_successes}",
            f"Security Score:     {result.overall_security_score:.1f}/100",
            "",
            "-" * 70,
            "INDIVIDUAL RESULTS",
            "-" * 70,
            "",
        ]

        for r in result.attack_results:
            status = "SUCCEEDED" if r.success else "BLOCKED"
            lines.append(f"[{r.severity.value.upper():8}] {r.attack_id}: {r.attack_name}")
            lines.append(f"           Status: {status}")
            lines.append(f"           Success Rate: {r.success_rate:.1%}")
            lines.append(f"           Detection Rate: {r.detection_rate:.1%}")
            lines.append(f"           Evasion Rate: {r.evasion_rate:.1%}")
            lines.append("")

        # Add recommendations
        lines.extend([
            "-" * 70,
            "RECOMMENDATIONS",
            "-" * 70,
            "",
        ])

        all_recommendations = []
        for r in result.attack_results:
            if r.success:  # Only show recommendations for successful attacks
                for rec in r.recommendations:
                    if rec not in all_recommendations:
                        all_recommendations.append(rec)

        if all_recommendations:
            for i, rec in enumerate(all_recommendations, 1):
                lines.append(f"{i}. {rec}")
        else:
            lines.append("No critical recommendations - all attacks blocked.")

        lines.extend([
            "",
            "=" * 70,
            "END OF REPORT",
            "=" * 70,
        ])

        return "\n".join(lines)

    def _generate_json_report(self, result: SuiteResult) -> str:
        """Generate JSON report."""
        report = {
            'suite_name': result.suite_name,
            'timestamp': datetime.fromtimestamp(result.start_time).isoformat(),
            'duration_seconds': result.duration_seconds,
            'summary': {
                'total_attacks': result.total_attacks,
                'successful_attacks': result.successful_attacks,
                'blocked_attacks': result.blocked_attacks,
                'critical_successes': result.critical_successes,
                'security_score': result.overall_security_score,
            },
            'attacks': [
                {
                    'attack_id': r.attack_id,
                    'attack_name': r.attack_name,
                    'severity': r.severity.value,
                    'success': r.success,
                    'success_rate': r.success_rate,
                    'detection_rate': r.detection_rate,
                    'evasion_rate': r.evasion_rate,
                    'details': r.details,
                    'recommendations': r.recommendations,
                }
                for r in result.attack_results
            ],
            'recommendations': list({
                rec
                for r in result.attack_results
                if r.success
                for rec in r.recommendations
            }),
        }
        return json.dumps(report, indent=2)


# Convenience functions

def run_quick_test(target: Any) -> SuiteResult:
    """
    Run a quick adversarial test.

    Args:
        target: Target to test

    Returns:
        SuiteResult from minimal suite
    """
    harness = RedTeamHarness(verbose=True)
    suite = AttackSuite.minimal()
    return harness.run_suite(suite, target)


def run_full_test(target: Any, detectors: Optional[Dict[str, Callable]] = None) -> SuiteResult:
    """
    Run full adversarial test suite.

    Args:
        target: Target to test
        detectors: Optional detector functions

    Returns:
        SuiteResult from full suite
    """
    harness = RedTeamHarness(verbose=True)
    suite = AttackSuite.full_suite()
    return harness.run_suite(suite, target, detectors)


__all__ = [
    'AttackConfiguration',
    'AttackSuite',
    'RedTeamHarness',
    'SuiteResult',
    'SuiteType',
    'run_quick_test',
    'run_full_test',
]
