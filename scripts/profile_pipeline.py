#!/usr/bin/env python3
"""Profiling and benchmarking script for the realtime pipeline.

This script:
1. Enables profiling in the server
2. Runs conformance harness tests under load
3. Collects performance metrics from Prometheus
4. Analyzes pyinstrument profiles
5. Generates a comprehensive performance report

Usage:
    # Profile with default settings
    python scripts/profile_pipeline.py

    # Profile with specific test fixtures
    python scripts/profile_pipeline.py --fixtures tests/realtime_harness/fixtures/audio_streaming.json

    # Profile with higher concurrency
    python scripts/profile_pipeline.py --concurrent-sessions 8

    # Generate comparison report
    python scripts/profile_pipeline.py --baseline-metrics baseline.json --output report.md
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def enable_profiling(config_file: Path = Path("unmute/main_websocket.py")) -> bool:
    """Enable profiling in the server by setting PROFILE_ACTIVE = True.

    Args:
        config_file: Path to main_websocket.py

    Returns:
        True if profiling was enabled, False if it was already enabled
    """
    content = config_file.read_text()

    if "PROFILE_ACTIVE = True" in content:
        print("✓ Profiling already enabled")
        return False

    # Replace PROFILE_ACTIVE = False with True
    new_content = content.replace("PROFILE_ACTIVE = False", "PROFILE_ACTIVE = True")

    if new_content == content:
        print("Warning: Could not find PROFILE_ACTIVE flag")
        return False

    config_file.write_text(new_content)
    print("✓ Enabled profiling in server")
    return True


def disable_profiling(config_file: Path = Path("unmute/main_websocket.py")):
    """Disable profiling in the server."""
    content = config_file.read_text()
    new_content = content.replace("PROFILE_ACTIVE = True", "PROFILE_ACTIVE = False")
    config_file.write_text(new_content)
    print("✓ Disabled profiling in server")


def start_server(port: int = 8000) -> subprocess.Popen:
    """Start the FastAPI server.

    Args:
        port: Port to run server on

    Returns:
        Process handle
    """
    print(f"Starting server on port {port}...")
    proc = subprocess.Popen(
        ["fastapi", "dev", "unmute/main_websocket.py", "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"http://localhost:{port}/v1/health", timeout=1)
            if response.status_code == 200:
                print(f"✓ Server ready (attempt {i+1}/{max_retries})")
                return proc
        except requests.exceptions.RequestException:
            time.sleep(1)

    raise RuntimeError("Server failed to start")


def collect_prometheus_metrics(port: int = 8000) -> dict[str, Any]:
    """Collect current Prometheus metrics from the server.

    Args:
        port: Server port

    Returns:
        Dict of metric name -> values
    """
    response = requests.get(f"http://localhost:{port}/metrics")
    response.raise_for_status()

    metrics = {}
    for line in response.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue

        parts = line.split()
        if len(parts) >= 2:
            metric_name = parts[0]
            try:
                metric_value = float(parts[1])
                metrics[metric_name] = metric_value
            except ValueError:
                pass

    return metrics


def run_harness_benchmarks(
    server_url: str,
    fixtures: list[Path] | None = None,
    concurrent_sessions: int = 1,
) -> dict[str, Any]:
    """Run conformance harness benchmarks.

    Args:
        server_url: URL of the server to test
        fixtures: Optional list of specific fixture files to run
        concurrent_sessions: Number of concurrent sessions

    Returns:
        Benchmark results dict
    """
    print(f"Running harness benchmarks with {concurrent_sessions} concurrent sessions...")

    cmd = [
        "python",
        "-m",
        "tests.realtime_harness.runner",
        "--url",
        server_url,
        "--benchmark",
    ]

    if fixtures:
        for fixture in fixtures:
            cmd.extend(["--fixture", str(fixture)])

    if concurrent_sessions > 1:
        cmd.extend(["--concurrent", str(concurrent_sessions)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running benchmarks:\n{result.stderr}")
        return {}

    # Parse benchmark results from output
    # The harness outputs JSON results to a file
    output_file = Path("benchmark_results.json")
    if output_file.exists():
        with open(output_file) as f:
            return json.load(f)

    return {}


def fetch_profile_html(port: int = 8000, output_file: Path = Path("profile.html")):
    """Fetch and save the pyinstrument profile HTML.

    Args:
        port: Server port
        output_file: Where to save the HTML profile
    """
    try:
        response = requests.get(f"http://localhost:{port}/profile")
        response.raise_for_status()

        output_file.write_text(response.text)
        print(f"✓ Saved profile to {output_file}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not fetch profile: {e}")
        return False


def analyze_metrics(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """Analyze metrics changes between before and after.

    Args:
        before: Metrics before test
        after: Metrics after test

    Returns:
        Dict of metric name -> {before, after, delta, delta_pct}
    """
    analysis = {}

    # Focus on key latency and throughput metrics
    key_metrics = [
        "worker_vllm_ttft_sum",
        "worker_vllm_ttft_count",
        "worker_stt_ttft_sum",
        "worker_stt_ttft_count",
        "worker_tts_ttft_sum",
        "worker_tts_ttft_count",
        "worker_vad_speech_duration_sum",
        "worker_vad_speech_duration_count",
        "unmute_span_duration_ms_sum",
        "unmute_span_duration_ms_count",
        "unmute_audio_ingestion_duration_ms_sum",
        "unmute_audio_ingestion_duration_ms_count",
        "unmute_websocket_send_duration_ms_sum",
        "unmute_websocket_send_duration_ms_count",
        "worker_output_queue_size",
        "worker_emit_queue_size",
    ]

    for metric in key_metrics:
        if metric in after:
            before_val = before.get(metric, 0)
            after_val = after[metric]
            delta = after_val - before_val

            analysis[metric] = {
                "before": before_val,
                "after": after_val,
                "delta": delta,
            }

            if before_val > 0:
                analysis[metric]["delta_pct"] = (delta / before_val) * 100

    return analysis


def generate_report(
    metrics_analysis: dict[str, dict[str, float]],
    benchmark_results: dict[str, Any],
    profile_file: Path | None,
    output_file: Path,
):
    """Generate a comprehensive performance report.

    Args:
        metrics_analysis: Analyzed metrics
        benchmark_results: Harness benchmark results
        profile_file: Path to profile HTML (if available)
        output_file: Where to save report
    """
    lines = ["# Realtime Pipeline Performance Report", ""]
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")

    # Calculate average latencies
    if "worker_vllm_ttft_sum" in metrics_analysis:
        vllm_sum = metrics_analysis["worker_vllm_ttft_sum"]["delta"]
        vllm_count = metrics_analysis.get("worker_vllm_ttft_count", {}).get("delta", 1)
        if vllm_count > 0:
            avg_vllm_ttft = (vllm_sum / vllm_count) * 1000  # Convert to ms
            lines.append(f"- **LLM TTFT**: {avg_vllm_ttft:.2f}ms average")

    if "worker_stt_ttft_sum" in metrics_analysis:
        stt_sum = metrics_analysis["worker_stt_ttft_sum"]["delta"]
        stt_count = metrics_analysis.get("worker_stt_ttft_count", {}).get("delta", 1)
        if stt_count > 0:
            avg_stt_ttft = (stt_sum / stt_count) * 1000
            lines.append(f"- **STT TTFT**: {avg_stt_ttft:.2f}ms average")

    if "worker_tts_ttft_sum" in metrics_analysis:
        tts_sum = metrics_analysis["worker_tts_ttft_sum"]["delta"]
        tts_count = metrics_analysis.get("worker_tts_ttft_count", {}).get("delta", 1)
        if tts_count > 0:
            avg_tts_ttft = (tts_sum / tts_count) * 1000
            lines.append(f"- **TTS TTFT**: {avg_tts_ttft:.2f}ms average")

    lines.append("")

    # Detailed Metrics
    lines.append("## Detailed Metrics")
    lines.append("")
    lines.append("| Metric | Before | After | Delta | % Change |")
    lines.append("|--------|--------|-------|-------|----------|")

    for metric, values in sorted(metrics_analysis.items()):
        delta_pct = values.get("delta_pct", 0)
        lines.append(
            f"| {metric} | {values['before']:.2f} | {values['after']:.2f} | "
            f"{values['delta']:.2f} | {delta_pct:.1f}% |"
        )

    lines.append("")

    # Benchmark Results
    if benchmark_results:
        lines.append("## Harness Benchmark Results")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(benchmark_results, indent=2))
        lines.append("```")
        lines.append("")

    # Profile Link
    if profile_file and profile_file.exists():
        lines.append("## Profiling Data")
        lines.append("")
        lines.append(f"Detailed pyinstrument profile: [{profile_file}]({profile_file})")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("Based on the profiling data:")
    lines.append("")

    # Check for queue buildup
    output_queue = metrics_analysis.get("worker_output_queue_size", {})
    if output_queue.get("after", 0) > 10:
        lines.append("- ⚠️  Output queue is building up - investigate backpressure")

    emit_queue = metrics_analysis.get("worker_emit_queue_size", {})
    if emit_queue.get("after", 0) > 10:
        lines.append("- ⚠️  Emit queue is building up - check WebSocket send performance")

    # Check for slow operations
    if "unmute_websocket_send_duration_ms_sum" in metrics_analysis:
        ws_sum = metrics_analysis["unmute_websocket_send_duration_ms_sum"]["delta"]
        ws_count = metrics_analysis.get(
            "unmute_websocket_send_duration_ms_count", {}
        ).get("delta", 1)
        if ws_count > 0:
            avg_ws_send = ws_sum / ws_count
            if avg_ws_send > 5:
                lines.append(
                    f"- ⚠️  WebSocket sends are slow ({avg_ws_send:.2f}ms avg) - check network"
                )

    if "unmute_audio_ingestion_duration_ms_sum" in metrics_analysis:
        audio_sum = metrics_analysis["unmute_audio_ingestion_duration_ms_sum"]["delta"]
        audio_count = metrics_analysis.get(
            "unmute_audio_ingestion_duration_ms_count", {}
        ).get("delta", 1)
        if audio_count > 0:
            avg_audio = audio_sum / audio_count
            if avg_audio > 10:
                lines.append(
                    f"- ⚠️  Opus decoding is slow ({avg_audio:.2f}ms avg) - may need optimization"
                )

    lines.append("")
    lines.append("---")
    lines.append(
        "*Generated by `scripts/profile_pipeline.py` - see that script for more options*"
    )

    output_file.write_text("\n".join(lines))
    print(f"✓ Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Profile the realtime pipeline")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run server on"
    )
    parser.add_argument(
        "--fixtures",
        nargs="+",
        type=Path,
        help="Specific fixture files to test",
    )
    parser.add_argument(
        "--concurrent-sessions",
        type=int,
        default=1,
        help="Number of concurrent test sessions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance_report.md"),
        help="Output report file",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        help="Baseline metrics JSON for comparison",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start server (assume already running)",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=Path("profile.html"),
        help="Where to save profile HTML",
    )

    args = parser.parse_args()

    try:
        # Enable profiling
        profiling_was_disabled = enable_profiling()

        # Start server if needed
        server_proc = None
        if not args.no_server:
            server_proc = start_server(args.port)

        # Collect baseline metrics
        print("Collecting baseline metrics...")
        time.sleep(2)  # Let server settle
        before_metrics = collect_prometheus_metrics(args.port)

        # Run benchmarks
        server_url = f"http://localhost:{args.port}"
        benchmark_results = run_harness_benchmarks(
            server_url,
            fixtures=args.fixtures,
            concurrent_sessions=args.concurrent_sessions,
        )

        # Collect final metrics
        print("Collecting final metrics...")
        time.sleep(1)
        after_metrics = collect_prometheus_metrics(args.port)

        # Fetch profile
        profile_fetched = fetch_profile_html(args.port, args.profile_output)

        # Analyze metrics
        metrics_analysis = analyze_metrics(before_metrics, after_metrics)

        # Generate report
        generate_report(
            metrics_analysis,
            benchmark_results,
            args.profile_output if profile_fetched else None,
            args.output,
        )

        print("\n✓ Profiling complete!")
        print(f"  Report: {args.output}")
        if profile_fetched:
            print(f"  Profile: {args.profile_output}")

        # Save metrics for future comparison
        metrics_file = Path("metrics_snapshot.json")
        with open(metrics_file, "w") as f:
            json.dump(
                {"before": before_metrics, "after": after_metrics, "analysis": metrics_analysis},
                f,
                indent=2,
            )
        print(f"  Metrics: {metrics_file}")

    finally:
        # Clean up
        if server_proc:
            print("\nStopping server...")
            server_proc.terminate()
            server_proc.wait(timeout=5)

        if profiling_was_disabled:
            disable_profiling()


if __name__ == "__main__":
    main()
