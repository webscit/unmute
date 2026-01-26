#!/usr/bin/env bash
# Run OpenAI Realtime API conformance harness locally
# This script reproduces CI harness runs for local debugging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/tests/realtime_harness/reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage information
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Run OpenAI Realtime API conformance harness locally (reproduces CI behavior)

OPTIONS:
    -m, --mode MODE         Test mode: conformance, fuzz, benchmark, all (default: all)
    -f, --fixture FIXTURE   Run specific fixture only
    -c, --category CATEGORY Filter by fixture category
    -o, --output OUTPUT     Custom output directory (default: tests/realtime_harness/reports)
    --no-server             Don't start server (connect to external instance)
    --host HOST             Server host (default: 127.0.0.1)
    --port PORT             Server port (default: 8765)
    --ttft-threshold MS     TTFT latency threshold in ms (default: 1000)
    --stt-threshold MS      STT flush threshold in ms (default: 300)
    --tool-rtt-threshold MS Tool call RTT threshold in ms (default: 2000)
    -h, --help              Show this help message

MODES:
    conformance    Run core conformance tests (event assertions, ordering, timing)
    fuzz           Run fuzzing tests (duplicate IDs, out-of-order events, corruption)
    benchmark      Run latency benchmarks (TTFT, STT flush, tool RTT, actuator RTT)
    all            Run all test modes (default)

EXAMPLES:
    # Run all test modes (CI default)
    $0

    # Run only conformance tests
    $0 --mode conformance

    # Run fuzz tests on specific fixture
    $0 --mode fuzz --fixture text_only_basic

    # Run benchmarks with custom thresholds
    $0 --mode benchmark --ttft-threshold 500 --stt-threshold 200

    # Run against external server
    $0 --no-server --host localhost --port 8000

    # Run specific category
    $0 --mode conformance --category audio_input

INTERPRETING RESULTS:
    - Reports are saved as timestamped JSON files in the reports directory
    - Exit code 0 = all tests passed
    - Exit code 1 = one or more tests failed
    - View detailed results in JSON artifacts or terminal output

DEBUGGING FAILURES:
    1. Check JSON report for failed fixtures and specific assertions
    2. Rerun failed fixture with: $0 --fixture <fixture_name>
    3. For server issues, run with --no-server and inspect logs manually
    4. Enable verbose output by modifying runner.py temporarily

EOF
}

# Parse arguments
MODE="all"
FIXTURE=""
CATEGORY=""
OUTPUT_DIR=""
NO_SERVER=""
HOST="127.0.0.1"
PORT="8765"
TTFT_THRESHOLD=""
STT_THRESHOLD=""
TOOL_RTT_THRESHOLD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -f|--fixture)
            FIXTURE="$2"
            shift 2
            ;;
        -c|--category)
            CATEGORY="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-server)
            NO_SERVER="--no-server"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --ttft-threshold)
            TTFT_THRESHOLD="$2"
            shift 2
            ;;
        --stt-threshold)
            STT_THRESHOLD="$2"
            shift 2
            ;;
        --tool-rtt-threshold)
            TOOL_RTT_THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$MODE" =~ ^(conformance|fuzz|benchmark|all)$ ]]; then
    echo -e "${RED}Invalid mode: $MODE${NC}"
    usage
    exit 1
fi

# Create reports directory
mkdir -p "$REPORTS_DIR"

# Use custom output dir if specified
if [[ -n "$OUTPUT_DIR" ]]; then
    REPORTS_DIR="$OUTPUT_DIR"
    mkdir -p "$REPORTS_DIR"
fi

# Timestamp for report filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Build base command
BASE_CMD="uv run python -m tests.realtime_harness.runner"
BASE_CMD="$BASE_CMD --host $HOST --port $PORT"

if [[ -n "$NO_SERVER" ]]; then
    BASE_CMD="$BASE_CMD $NO_SERVER"
fi

if [[ -n "$FIXTURE" ]]; then
    BASE_CMD="$BASE_CMD --fixture $FIXTURE"
fi

if [[ -n "$CATEGORY" ]]; then
    BASE_CMD="$BASE_CMD --category $CATEGORY"
fi

# Run tests based on mode
run_tests() {
    local mode=$1
    local output_file="$REPORTS_DIR/${mode}_${TIMESTAMP}.json"
    local cmd="$BASE_CMD --output $output_file"

    case $mode in
        conformance)
            echo -e "${BLUE}=== Running Conformance Tests ===${NC}"
            ;;
        fuzz)
            echo -e "${BLUE}=== Running Fuzz Tests ===${NC}"
            cmd="$cmd --fuzz"
            ;;
        benchmark)
            echo -e "${BLUE}=== Running Latency Benchmarks ===${NC}"
            cmd="$cmd --benchmark"
            if [[ -n "$TTFT_THRESHOLD" ]]; then
                cmd="$cmd --ttft-threshold $TTFT_THRESHOLD"
            fi
            if [[ -n "$STT_THRESHOLD" ]]; then
                cmd="$cmd --stt-threshold $STT_THRESHOLD"
            fi
            if [[ -n "$TOOL_RTT_THRESHOLD" ]]; then
                cmd="$cmd --tool-rtt-threshold $TOOL_RTT_THRESHOLD"
            fi
            ;;
    esac

    echo -e "${YELLOW}Command: $cmd${NC}"
    echo ""

    if $cmd; then
        echo -e "${GREEN}✓ $mode tests PASSED${NC}"
        echo -e "${GREEN}Report saved to: $output_file${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ $mode tests FAILED${NC}"
        echo -e "${RED}Report saved to: $output_file${NC}"
        echo ""
        return 1
    fi
}

# Track overall success
OVERALL_SUCCESS=0

# Run requested modes
if [[ "$MODE" == "all" ]]; then
    run_tests "conformance" || OVERALL_SUCCESS=1
    run_tests "fuzz" || OVERALL_SUCCESS=1
    run_tests "benchmark" || OVERALL_SUCCESS=1
else
    run_tests "$MODE" || OVERALL_SUCCESS=1
fi

# Summary
echo -e "${BLUE}=== Summary ===${NC}"
if [[ $OVERALL_SUCCESS -eq 0 ]]; then
    echo -e "${GREEN}All tests PASSED ✓${NC}"
else
    echo -e "${RED}Some tests FAILED ✗${NC}"
fi

echo -e "Reports directory: ${YELLOW}$REPORTS_DIR${NC}"

exit $OVERALL_SUCCESS
