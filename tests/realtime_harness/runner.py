"""Conformance harness runner for OpenAI Realtime API.

This module provides the main runner that launches a FastAPI server, replays
fixtures, validates responses, and generates reports.
"""
# pyright: reportPrivateUsage=false

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from tests.realtime_harness.benchmarks import (
    BenchmarkReport,
    BenchmarkRunner,
    BenchmarkThresholds,
)
from tests.realtime_harness.fixture_parser import FixtureParser
from tests.realtime_harness.fixture_schema import (
    FixtureEventType,
    FixtureResult,
    HarnessReport,
)
from tests.realtime_harness.fuzz_generators import (
    FuzzCampaign,
    get_predefined_edge_cases,
)
from tests.realtime_harness.validators import ProtocolValidator
from tests.realtime_harness.websocket_replayer import WebSocketReplayer

console = Console()


class ServerLauncher:
    """Manages launching and health-checking the FastAPI server."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        module: str = "unmute.main_websocket:app",
    ):
        """Initialize server launcher.

        Args:
            host: Host to bind server to
            port: Port to bind server to
            module: Python module path to FastAPI app
        """
        self.host = host
        self.port = port
        self.module = module
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

    def start(self, timeout: int = 30) -> None:
        """Start the server and wait for it to be ready.

        Args:
            timeout: Seconds to wait for server to be ready

        Raises:
            RuntimeError: If server fails to start
        """
        console.print(f"[yellow]Starting server on {self.host}:{self.port}...[/yellow]")

        # Launch uvicorn in subprocess
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                self.module,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--ws-per-message-deflate=false",
                "--log-level",
                "warning",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        start = time.time()
        while time.time() - start < timeout:
            if self._is_ready():
                console.print(f"[green]Server ready at {self.base_url}[/green]")
                return

            # Check if process died
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode() if self.process.stderr else ""
                raise RuntimeError(f"Server process died: {stderr}")

            time.sleep(0.5)

        self.stop()
        raise RuntimeError(f"Server failed to start within {timeout} seconds")

    def _is_ready(self) -> bool:
        """Check if server is ready by polling health endpoint."""
        try:
            response = httpx.get(f"{self.base_url}/v1/health", timeout=1.0)
            return response.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        """Stop the server gracefully."""
        if self.process:
            console.print("[yellow]Stopping server...[/yellow]")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            console.print("[green]Server stopped[/green]")


class HarnessRunner:
    """Main harness runner."""

    def __init__(
        self,
        fixtures_dir: Optional[Path] = None,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
    ):
        """Initialize harness runner.

        Args:
            fixtures_dir: Directory containing fixtures
            server_host: Host for server
            server_port: Port for server
        """
        self.parser = FixtureParser(fixtures_dir=fixtures_dir)
        self.server = ServerLauncher(host=server_host, port=server_port)
        self.ws_base_url = f"ws://{server_host}:{server_port}"

    async def run_fixture(self, fixture_name: str) -> FixtureResult:
        """Run a single fixture and return result.

        Args:
            fixture_name: Name of fixture to run

        Returns:
            Fixture result with validation details
        """
        start_time = time.time()

        try:
            # Load fixture
            fixture = self.parser.load_fixture(fixture_name)

            # Replay fixture
            replayer = WebSocketReplayer(base_url=self.ws_base_url)
            trace = await replayer.replay_fixture(fixture)

            # Validate trace
            validator = ProtocolValidator(fixture, trace)
            validation_result = validator.validate_all()

            # Build result
            result = FixtureResult(
                fixture_name=fixture.metadata.name,
                passed=validation_result.passed,
                duration_ms=(time.time() - start_time) * 1000,
                assertions_passed=validation_result.passed_assertions,
                assertions_failed=validation_result.failed_assertions,
                failures=validation_result.failure_messages,
                warnings=validation_result.warning_messages,
                server_events_received=len(trace.received_events),
            )

            # Add trace errors as failures
            if trace.errors:
                result.passed = False
                result.failures.extend(trace.errors)

            return result

        except Exception as e:
            return FixtureResult(
                fixture_name=fixture_name,
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                assertions_passed=0,
                assertions_failed=0,
                failures=[f"Fixture execution error: {str(e)}"],
            )

    async def run_all_fixtures(
        self,
        category: Optional[FixtureEventType] = None,
        tags: Optional[list[str]] = None,
        fixture_names: Optional[list[str]] = None,
    ) -> HarnessReport:
        """Run all fixtures and generate report.

        Args:
            category: Optional filter by category
            tags: Optional filter by tags
            fixture_names: Optional list of specific fixture names to run

        Returns:
            Complete harness report
        """
        start_time = time.time()

        # Load fixtures
        if fixture_names:
            fixtures = [self.parser.load_fixture(name) for name in fixture_names]
        else:
            fixtures = self.parser.load_all_fixtures(category=category, tags=tags)

        if not fixtures:
            console.print("[yellow]No fixtures found matching criteria[/yellow]")
            return HarnessReport(
                timestamp=datetime.now().isoformat(),
                total_fixtures=0,
                fixtures_passed=0,
                fixtures_failed=0,
                total_duration_ms=0,
                fixture_results=[],
            )

        console.print(f"\n[bold]Running {len(fixtures)} fixture(s)...[/bold]\n")

        # Run fixtures with progress bar
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for fixture in fixtures:
                task = progress.add_task(
                    f"Running {fixture.metadata.name}...", total=None
                )

                result = await self.run_fixture(fixture.metadata.name)
                results.append(result)

                status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
                progress.update(
                    task,
                    description=f"{status} {fixture.metadata.name} ({result.duration_ms:.0f}ms)",
                    completed=True,
                )

        # Generate report
        total_duration_ms = (time.time() - start_time) * 1000
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        report = HarnessReport(
            timestamp=datetime.now().isoformat(),
            total_fixtures=len(results),
            fixtures_passed=passed_count,
            fixtures_failed=failed_count,
            total_duration_ms=total_duration_ms,
            fixture_results=results,
            environment={
                "server_url": self.ws_base_url,
                "python_version": sys.version,
            },
        )

        return report

    async def run_fuzz_campaign(
        self,
        base_fixtures: Optional[list[str]] = None,
        strategies: Optional[list[str]] = None,
        include_edge_cases: bool = True,
    ) -> HarnessReport:
        """Run fuzz campaign on base fixtures.

        Args:
            base_fixtures: List of base fixture names to fuzz (None = all)
            strategies: List of fuzz strategy names to apply (None = all)
            include_edge_cases: Whether to include predefined edge cases

        Returns:
            Harness report with fuzz test results
        """
        start_time = time.time()

        # Load base fixtures
        if base_fixtures:
            fixtures = [self.parser.load_fixture(name) for name in base_fixtures]
        else:
            fixtures = self.parser.load_all_fixtures()

        # Generate fuzzed fixtures
        campaign = FuzzCampaign()
        fuzzed_fixtures = campaign.generate_fuzzed_fixtures(fixtures, strategies)

        # Add edge cases
        if include_edge_cases:
            fuzzed_fixtures.extend(get_predefined_edge_cases())

        if not fuzzed_fixtures:
            console.print("[yellow]No fuzzed fixtures generated[/yellow]")
            return HarnessReport(
                timestamp=datetime.now().isoformat(),
                total_fixtures=0,
                fixtures_passed=0,
                fixtures_failed=0,
                total_duration_ms=0,
                fixture_results=[],
            )

        console.print(f"\n[bold]Running {len(fuzzed_fixtures)} fuzzed fixture(s)...[/bold]\n")

        # Run fuzzed fixtures
        results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for fixture in fuzzed_fixtures:
                task = progress.add_task(
                    f"Running {fixture.metadata.name}...", total=None
                )

                # Replay fixture
                try:
                    replayer = WebSocketReplayer(base_url=self.ws_base_url)
                    trace = await replayer.replay_fixture(fixture)

                    # Validate trace
                    validator = ProtocolValidator(fixture, trace)
                    validation_result = validator.validate_all()

                    result = FixtureResult(
                        fixture_name=fixture.metadata.name,
                        passed=validation_result.passed,
                        duration_ms=(time.time() - start_time) * 1000,
                        assertions_passed=validation_result.passed_assertions,
                        assertions_failed=validation_result.failed_assertions,
                        failures=validation_result.failure_messages,
                        warnings=validation_result.warning_messages,
                        server_events_received=len(trace.received_events),
                    )

                    if trace.errors:
                        result.passed = False
                        result.failures.extend(trace.errors)

                except Exception as e:
                    result = FixtureResult(
                        fixture_name=fixture.metadata.name,
                        passed=False,
                        duration_ms=0,
                        assertions_passed=0,
                        assertions_failed=0,
                        failures=[f"Fuzz fixture error: {str(e)}"],
                    )

                results.append(result)

                status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
                progress.update(
                    task,
                    description=f"{status} {fixture.metadata.name} ({result.duration_ms:.0f}ms)",
                    completed=True,
                )

        # Generate report
        total_duration_ms = (time.time() - start_time) * 1000
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        report = HarnessReport(
            timestamp=datetime.now().isoformat(),
            total_fixtures=len(results),
            fixtures_passed=passed_count,
            fixtures_failed=failed_count,
            total_duration_ms=total_duration_ms,
            fixture_results=results,
            environment={
                "server_url": self.ws_base_url,
                "python_version": sys.version,
                "mode": "fuzz",
            },
        )

        return report

    async def run_benchmarks(
        self,
        fixture_names: Optional[list[str]] = None,
        thresholds: Optional[BenchmarkThresholds] = None,
    ) -> BenchmarkReport:
        """Run latency benchmarks on fixtures.

        Args:
            fixture_names: List of fixture names to benchmark (None = all)
            thresholds: Optional custom latency thresholds

        Returns:
            Benchmark report with latency measurements
        """
        # Load fixtures
        if fixture_names:
            fixtures = [self.parser.load_fixture(name) for name in fixture_names]
        else:
            fixtures = self.parser.load_all_fixtures()

        if not fixtures:
            console.print("[yellow]No fixtures found for benchmarking[/yellow]")
            return BenchmarkReport(
                timestamp=datetime.now().isoformat(),
                total_fixtures=0,
                benchmark_results=[],
                thresholds=thresholds or BenchmarkThresholds(),
            )

        console.print(f"\n[bold]Benchmarking {len(fixtures)} fixture(s)...[/bold]\n")

        # Collect traces
        traces = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for fixture in fixtures:
                task = progress.add_task(
                    f"Replaying {fixture.metadata.name}...", total=None
                )

                try:
                    replayer = WebSocketReplayer(base_url=self.ws_base_url)
                    trace = await replayer.replay_fixture(fixture)
                    traces.append((trace, fixture.metadata.name))
                except Exception as e:
                    console.print(f"[red]Error replaying {fixture.metadata.name}: {e}[/red]")

                progress.update(task, completed=True)

        # Run benchmarks
        runner = BenchmarkRunner(thresholds=thresholds)
        report = runner.run_benchmarks(traces)

        return report

    def run(
        self,
        category: Optional[FixtureEventType] = None,
        tags: Optional[list[str]] = None,
        fixture_names: Optional[list[str]] = None,
        output_file: Optional[Path] = None,
        start_server: bool = True,
    ) -> HarnessReport:
        """Run the harness with optional server management.

        Args:
            category: Optional filter by category
            tags: Optional filter by tags
            fixture_names: Optional specific fixtures to run
            output_file: Optional file to write JSON report
            start_server: Whether to start/stop server (False for external server)

        Returns:
            Harness report
        """
        try:
            # Start server if requested
            if start_server:
                self.server.start()

            # Run fixtures
            report = asyncio.run(
                self.run_all_fixtures(
                    category=category, tags=tags, fixture_names=fixture_names
                )
            )

            # Display results
            self._display_report(report)

            # Write JSON report if requested
            if output_file:
                output_file.write_text(json.dumps(report.model_dump(), indent=2))
                console.print(f"\n[green]Report written to {output_file}[/green]")

            return report

        finally:
            # Stop server if we started it
            if start_server:
                self.server.stop()

    def _display_report(self, report: HarnessReport) -> None:
        """Display report in terminal.

        Args:
            report: Report to display
        """
        console.print("\n[bold]Conformance Harness Report[/bold]")
        console.print(f"Timestamp: {report.timestamp}")
        console.print(f"Duration: {report.total_duration_ms:.0f}ms\n")

        # Summary table
        summary = Table(title="Summary")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="magenta")

        summary.add_row("Total Fixtures", str(report.total_fixtures))
        summary.add_row(
            "Passed", f"[green]{report.fixtures_passed}[/green]"
        )
        summary.add_row(
            "Failed", f"[red]{report.fixtures_failed}[/red]"
        )

        console.print(summary)

        # Failed fixtures details
        if report.fixtures_failed > 0:
            console.print("\n[bold red]Failed Fixtures:[/bold red]")
            for result in report.fixture_results:
                if not result.passed:
                    console.print(f"\n[red]✗ {result.fixture_name}[/red]")
                    for failure in result.failures:
                        console.print(f"  - {failure}")

        # Success message
        if report.fixtures_failed == 0:
            console.print("\n[bold green]All fixtures passed! ✓[/bold green]")
        else:
            console.print(
                f"\n[bold red]{report.fixtures_failed} fixture(s) failed[/bold red]"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI Realtime API Conformance Harness"
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fuzz",
        action="store_true",
        help="Run fuzz campaign to test error handling",
    )
    mode_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmarks",
    )

    # Fixture selection
    parser.add_argument(
        "--fixture", "-f", action="append", help="Specific fixture(s) to run"
    )
    parser.add_argument(
        "--category", "-c", type=FixtureEventType, help="Filter by category"
    )
    parser.add_argument("--tag", "-t", action="append", help="Filter by tag(s)")

    # Output
    parser.add_argument("--output", "-o", type=Path, help="Output JSON report file")

    # Server config
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start server (use external server)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    # Fuzz options
    parser.add_argument(
        "--fuzz-strategy",
        action="append",
        choices=["duplicate_ids", "out_of_order", "invalid_tool_output", "malformed_payload"],
        help="Specific fuzz strategy to apply (can be used multiple times)",
    )
    parser.add_argument(
        "--no-edge-cases",
        action="store_true",
        help="Don't include predefined edge cases in fuzz campaign",
    )

    # Benchmark options
    parser.add_argument(
        "--ttft-threshold",
        type=int,
        default=1000,
        help="TTFT threshold in milliseconds (default: 1000)",
    )
    parser.add_argument(
        "--stt-threshold",
        type=int,
        default=300,
        help="STT flush latency threshold in milliseconds (default: 300)",
    )
    parser.add_argument(
        "--tool-rtt-threshold",
        type=int,
        default=2000,
        help="Tool call RTT threshold in milliseconds (default: 2000)",
    )
    parser.add_argument(
        "--actuator-rtt-threshold",
        type=int,
        default=500,
        help="Actuator RTT threshold in milliseconds (default: 500)",
    )

    args = parser.parse_args()

    runner = HarnessRunner(server_host=args.host, server_port=args.port)

    try:
        # Start server if requested
        if not args.no_server:
            runner.server.start()

        # Run based on mode
        if args.fuzz:
            # Fuzz mode
            report = asyncio.run(
                runner.run_fuzz_campaign(
                    base_fixtures=args.fixture,
                    strategies=args.fuzz_strategy,
                    include_edge_cases=not args.no_edge_cases,
                )
            )
            runner._display_report(report)

            if args.output:
                args.output.write_text(json.dumps(report.model_dump(), indent=2))
                console.print(f"\n[green]Report written to {args.output}[/green]")

            sys.exit(0 if report.fixtures_failed == 0 else 1)

        elif args.benchmark:
            # Benchmark mode
            thresholds = BenchmarkThresholds(
                ttft_ms=args.ttft_threshold,
                stt_flush_ms=args.stt_threshold,
                tool_call_rtt_ms=args.tool_rtt_threshold,
                actuator_rtt_ms=args.actuator_rtt_threshold,
            )

            report = asyncio.run(
                runner.run_benchmarks(
                    fixture_names=args.fixture,
                    thresholds=thresholds,
                )
            )

            # Display benchmark report
            benchmark_runner = BenchmarkRunner(thresholds=thresholds)
            console.print(benchmark_runner.format_report(report))

            if args.output:
                benchmark_runner.save_json_report(report, str(args.output))
                console.print(f"\n[green]Report written to {args.output}[/green]")

            # Exit with failure if any measurements failed
            total_failed = sum(r.failed_measurements for r in report.benchmark_results)
            sys.exit(0 if total_failed == 0 else 1)

        else:
            # Normal conformance mode
            report = runner.run(
                category=args.category,
                tags=args.tag,
                fixture_names=args.fixture,
                output_file=args.output,
                start_server=False,  # Already started above if needed
            )
            sys.exit(0 if report.fixtures_failed == 0 else 1)

    finally:
        # Stop server if we started it
        if not args.no_server:
            runner.server.stop()
