"""Latency benchmarking for conformance harness.

This module provides latency measurement capabilities for:
- TTFT (Time To First Token)
- STT flush latency
- Tool call Round-Trip Time (RTT)
- Actuator RTT

Benchmarks enforce configurable thresholds and generate machine-readable reports.
"""

import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .websocket_replayer import ReplayTrace


class BenchmarkThresholds(BaseModel):
    """Configurable latency thresholds for benchmarks."""

    ttft_ms: Optional[int] = Field(
        1000, description="Time to first token threshold (ms)"
    )
    stt_flush_ms: Optional[int] = Field(
        300, description="STT flush latency threshold (ms)"
    )
    tool_call_rtt_ms: Optional[int] = Field(
        2000, description="Tool call round-trip time threshold (ms)"
    )
    actuator_rtt_ms: Optional[int] = Field(
        500, description="Actuator command RTT threshold (ms)"
    )


class LatencyMeasurement(BaseModel):
    """Single latency measurement."""

    name: str = Field(..., description="Measurement name")
    latency_ms: float = Field(..., description="Measured latency in milliseconds")
    threshold_ms: Optional[int] = Field(None, description="Configured threshold")
    passed: bool = Field(..., description="Whether measurement passed threshold")
    event_a: str = Field(..., description="Starting event type")
    event_b: str = Field(..., description="Ending event type")
    timestamp_a: float = Field(..., description="Starting event timestamp")
    timestamp_b: float = Field(..., description="Ending event timestamp")


class BenchmarkResult(BaseModel):
    """Result of running benchmarks on a trace."""

    fixture_name: str
    measurements: list[LatencyMeasurement] = Field(default_factory=list)
    total_measurements: int = 0
    passed_measurements: int = 0
    failed_measurements: int = 0
    summary_statistics: dict[str, dict[str, float]] = Field(default_factory=dict)


class BenchmarkReport(BaseModel):
    """Overall benchmark report across multiple traces."""

    timestamp: str
    total_fixtures: int
    benchmark_results: list[BenchmarkResult]
    thresholds: BenchmarkThresholds
    environment: dict[str, str] = Field(default_factory=dict)


@dataclass
class EventPair:
    """Pair of events for latency calculation."""

    event_a_type: str
    event_b_type: str
    name: str
    description: str


class LatencyBenchmarker:
    """Measures latency between events in replay traces."""

    # Standard benchmark event pairs
    TTFT_PAIR = EventPair(
        event_a_type="response.created",
        event_b_type="response.audio.delta",  # First audio/text delta
        name="TTFT",
        description="Time to first token (response created to first content delta)",
    )

    TTFT_TEXT_PAIR = EventPair(
        event_a_type="response.created",
        event_b_type="response.text.delta",
        name="TTFT_text",
        description="Time to first text token",
    )

    STT_FLUSH_PAIR = EventPair(
        event_a_type="input_audio_buffer.committed",
        event_b_type="conversation.item.input_audio_transcription.completed",
        name="STT_flush",
        description="STT flush latency (buffer commit to transcription complete)",
    )

    TOOL_CALL_PAIR = EventPair(
        event_a_type="response.function_call_arguments.done",
        event_b_type="conversation.item.created",  # Server receives function output
        name="tool_call_RTT",
        description="Tool call round-trip time",
    )

    def __init__(self, thresholds: Optional[BenchmarkThresholds] = None):
        self.thresholds = thresholds or BenchmarkThresholds()

    def measure_latency(
        self,
        trace: ReplayTrace,
        event_a_type: str,
        event_b_type: str,
        name: str,
        threshold_ms: Optional[int],
    ) -> Optional[LatencyMeasurement]:
        """Measure latency between first occurrence of two event types.

        Args:
            trace: Replay trace containing events
            event_a_type: Starting event type
            event_b_type: Ending event type
            name: Measurement name
            threshold_ms: Optional threshold for pass/fail

        Returns:
            LatencyMeasurement or None if events not found
        """
        # Find first occurrence of each event
        event_a = trace.get_first_event(event_a_type)
        event_b = trace.get_first_event(event_b_type)

        if not event_a or not event_b:
            return None

        # Calculate latency
        latency_ms = (event_b.timestamp - event_a.timestamp) * 1000

        # Check against threshold
        passed = True
        if threshold_ms is not None:
            passed = latency_ms <= threshold_ms

        return LatencyMeasurement(
            name=name,
            latency_ms=latency_ms,
            threshold_ms=threshold_ms,
            passed=passed,
            event_a=event_a_type,
            event_b=event_b_type,
            timestamp_a=event_a.timestamp,
            timestamp_b=event_b.timestamp,
        )

    def measure_ttft(self, trace: ReplayTrace) -> Optional[LatencyMeasurement]:
        """Measure Time To First Token."""
        # Try audio delta first, fall back to text delta
        result = self.measure_latency(
            trace,
            self.TTFT_PAIR.event_a_type,
            self.TTFT_PAIR.event_b_type,
            self.TTFT_PAIR.name,
            self.thresholds.ttft_ms,
        )
        if result:
            return result

        return self.measure_latency(
            trace,
            self.TTFT_TEXT_PAIR.event_a_type,
            self.TTFT_TEXT_PAIR.event_b_type,
            self.TTFT_TEXT_PAIR.name,
            self.thresholds.ttft_ms,
        )

    def measure_stt_flush(self, trace: ReplayTrace) -> Optional[LatencyMeasurement]:
        """Measure STT flush latency."""
        return self.measure_latency(
            trace,
            self.STT_FLUSH_PAIR.event_a_type,
            self.STT_FLUSH_PAIR.event_b_type,
            self.STT_FLUSH_PAIR.name,
            self.thresholds.stt_flush_ms,
        )

    def measure_tool_call_rtt(
        self, trace: ReplayTrace
    ) -> Optional[LatencyMeasurement]:
        """Measure tool call round-trip time."""
        return self.measure_latency(
            trace,
            self.TOOL_CALL_PAIR.event_a_type,
            self.TOOL_CALL_PAIR.event_b_type,
            self.TOOL_CALL_PAIR.name,
            self.thresholds.tool_call_rtt_ms,
        )

    def measure_actuator_rtt(self, trace: ReplayTrace) -> Optional[LatencyMeasurement]:
        """Measure actuator command round-trip time.

        Note: This requires specific actuator-related events in the trace.
        Currently returns None as actuator events are not yet standardized.
        """
        # TODO: Define actuator event types once schema is finalized
        # For now, look for generic pattern
        actuator_events = [
            e for e in trace.received_events if "actuator" in e.event_type.lower()
        ]
        if len(actuator_events) >= 2:
            return self.measure_latency(
                trace,
                actuator_events[0].event_type,
                actuator_events[-1].event_type,
                "actuator_RTT",
                self.thresholds.actuator_rtt_ms,
            )
        return None

    def benchmark_trace(
        self, trace: ReplayTrace, fixture_name: str
    ) -> BenchmarkResult:
        """Run all applicable benchmarks on a trace.

        Args:
            trace: Replay trace to benchmark
            fixture_name: Name of the fixture being benchmarked

        Returns:
            BenchmarkResult with all measurements
        """
        measurements = []

        # Run all benchmark types
        benchmark_methods = [
            self.measure_ttft,
            self.measure_stt_flush,
            self.measure_tool_call_rtt,
            self.measure_actuator_rtt,
        ]

        for method in benchmark_methods:
            try:
                measurement = method(trace)
                if measurement:
                    measurements.append(measurement)
            except Exception as e:
                # Skip measurements that fail
                print(f"Warning: Benchmark {method.__name__} failed: {e}")

        # Calculate statistics
        passed = sum(1 for m in measurements if m.passed)
        failed = len(measurements) - passed

        # Group measurements by name for statistics
        stats: dict[str, dict[str, float]] = {}
        measurement_groups: dict[str, list[float]] = {}

        for m in measurements:
            if m.name not in measurement_groups:
                measurement_groups[m.name] = []
            measurement_groups[m.name].append(m.latency_ms)

        for name, values in measurement_groups.items():
            if len(values) > 0:
                stats[name] = {
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                }
                if len(values) > 1:
                    stats[name]["stddev"] = statistics.stdev(values)

        return BenchmarkResult(
            fixture_name=fixture_name,
            measurements=measurements,
            total_measurements=len(measurements),
            passed_measurements=passed,
            failed_measurements=failed,
            summary_statistics=stats,
        )


class BenchmarkRunner:
    """Runs benchmarks across multiple fixtures and generates reports."""

    def __init__(self, thresholds: Optional[BenchmarkThresholds] = None):
        self.thresholds = thresholds or BenchmarkThresholds()
        self.benchmarker = LatencyBenchmarker(thresholds=self.thresholds)

    def run_benchmarks(
        self, traces: list[tuple[ReplayTrace, str]]
    ) -> BenchmarkReport:
        """Run benchmarks on multiple traces.

        Args:
            traces: List of (trace, fixture_name) tuples

        Returns:
            BenchmarkReport with aggregated results
        """
        results = []

        for trace, fixture_name in traces:
            result = self.benchmarker.benchmark_trace(trace, fixture_name)
            results.append(result)

        import platform
        import sys

        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_fixtures=len(traces),
            benchmark_results=results,
            thresholds=self.thresholds,
            environment={
                "python_version": sys.version,
                "platform": platform.platform(),
            },
        )

    def format_report(self, report: BenchmarkReport) -> str:
        """Format benchmark report for terminal display.

        Args:
            report: BenchmarkReport to format

        Returns:
            Formatted string for terminal output
        """
        lines = []
        lines.append("=" * 80)
        lines.append("LATENCY BENCHMARK REPORT")
        lines.append("=" * 80)
        lines.append(f"Timestamp: {report.timestamp}")
        lines.append(f"Total fixtures: {report.total_fixtures}")
        lines.append("")

        lines.append("Thresholds:")
        lines.append(f"  TTFT: {report.thresholds.ttft_ms}ms")
        lines.append(f"  STT flush: {report.thresholds.stt_flush_ms}ms")
        lines.append(f"  Tool call RTT: {report.thresholds.tool_call_rtt_ms}ms")
        lines.append(f"  Actuator RTT: {report.thresholds.actuator_rtt_ms}ms")
        lines.append("")

        # Summary table
        total_passed = sum(r.passed_measurements for r in report.benchmark_results)
        total_failed = sum(r.failed_measurements for r in report.benchmark_results)
        total_measurements = total_passed + total_failed

        lines.append(f"Summary: {total_passed}/{total_measurements} passed")
        lines.append("")

        # Per-fixture results
        for result in report.benchmark_results:
            lines.append(f"Fixture: {result.fixture_name}")
            lines.append(
                f"  Measurements: {result.passed_measurements}/{result.total_measurements} passed"
            )

            for measurement in result.measurements:
                status = "✓" if measurement.passed else "✗"
                threshold_str = (
                    f" (threshold: {measurement.threshold_ms}ms)"
                    if measurement.threshold_ms
                    else ""
                )
                lines.append(
                    f"    {status} {measurement.name}: {measurement.latency_ms:.2f}ms{threshold_str}"
                )

            # Statistics
            if result.summary_statistics:
                lines.append("  Statistics:")
                for name, stats in result.summary_statistics.items():
                    lines.append(f"    {name}:")
                    lines.append(f"      mean: {stats['mean']:.2f}ms")
                    lines.append(f"      min: {stats['min']:.2f}ms")
                    lines.append(f"      max: {stats['max']:.2f}ms")
                    lines.append(f"      median: {stats['median']:.2f}ms")
                    if "stddev" in stats:
                        lines.append(f"      stddev: {stats['stddev']:.2f}ms")

            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def save_json_report(self, report: BenchmarkReport, output_path: str):
        """Save benchmark report as JSON.

        Args:
            report: BenchmarkReport to save
            output_path: Path to output JSON file
        """
        import json

        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)
