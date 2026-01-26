"""Conformance harness runner for OpenAI Realtime API.

This module provides the main runner that launches a FastAPI server, replays
fixtures, validates responses, and generates reports.
"""

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

from tests.realtime_harness.fixture_parser import FixtureParser
from tests.realtime_harness.fixture_schema import (
    FixtureEventType,
    FixtureResult,
    HarnessReport,
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
    parser.add_argument(
        "--fixture", "-f", action="append", help="Specific fixture(s) to run"
    )
    parser.add_argument(
        "--category", "-c", type=FixtureEventType, help="Filter by category"
    )
    parser.add_argument("--tag", "-t", action="append", help="Filter by tag(s)")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON report file")
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start server (use external server)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")

    args = parser.parse_args()

    runner = HarnessRunner(server_host=args.host, server_port=args.port)

    report = runner.run(
        category=args.category,
        tags=args.tag,
        fixture_names=args.fixture,
        output_file=args.output,
        start_server=not args.no_server,
    )

    # Exit with failure code if any fixtures failed
    sys.exit(0 if report.fixtures_failed == 0 else 1)
