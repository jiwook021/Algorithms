#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_builds.sh — Discover and build/test every Makefile-based project
#
# Usage:
#   ./run_all_builds.sh [ROOT_DIR] [--build-only|--test-only]
#
# Output: Tab-separated report lines to stdout
#   DIR \t BUILD_STATUS \t TEST_STATUS \t ERROR_SNIPPET
#
# Statuses: PASS, FAIL, SKIP (no target), N/A
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${1:-$(pwd)}"
MODE="${2:-all}"  # all | --build-only | --test-only

REPORT_FILE=$(mktemp /tmp/build_test_report.XXXXXX)
FAIL_LOG_DIR=$(mktemp -d /tmp/build_test_failures.XXXXXX)

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BUILD_PASS=0
BUILD_FAIL=0
BUILD_SKIP=0
TEST_PASS=0
TEST_FAIL=0
TEST_SKIP=0
TOTAL=0

# Detect available test target in a Makefile
detect_test_target() {
    local makefile="$1"
    # Priority order: test, run_test, then any *_test target
    if grep -qE '^test\s*:' "$makefile" 2>/dev/null; then
        echo "test"
    elif grep -qE '^run_test\s*:' "$makefile" 2>/dev/null; then
        echo "run_test"
    else
        local custom
        custom=$(grep -oE '^[a-zA-Z_]*test[a-zA-Z_]*\s*:' "$makefile" 2>/dev/null | head -1 | sed 's/\s*://')
        if [[ -n "$custom" ]]; then
            echo "$custom"
        else
            echo ""
        fi
    fi
}

# Detect build target
detect_build_target() {
    local makefile="$1"
    if grep -qE '^main\s*:' "$makefile" 2>/dev/null; then
        echo "main"
    elif grep -qE '^all\s*:' "$makefile" 2>/dev/null; then
        echo "all"
    else
        echo "main"  # default
    fi
}

echo -e "${CYAN}=== Build & Test Runner ===${NC}"
echo -e "${CYAN}Root: $ROOT${NC}"
echo ""

# Find all directories containing a Makefile (skip .git, node_modules, etc.)
while IFS= read -r makefile; do
    dir=$(dirname "$makefile")
    rel_dir="${dir#"$ROOT"/}"
    TOTAL=$((TOTAL + 1))

    build_status="N/A"
    test_status="N/A"
    error_snippet=""

    # --- BUILD ---
    if [[ "$MODE" != "--test-only" ]]; then
        build_target=$(detect_build_target "$makefile")

        # Clean first to ensure fresh build
        make -C "$dir" clean >/dev/null 2>&1 || true

        build_log="$FAIL_LOG_DIR/${rel_dir//\//_}_build.log"
        if make -C "$dir" "$build_target" >"$build_log" 2>&1; then
            build_status="PASS"
            BUILD_PASS=$((BUILD_PASS + 1))
            rm -f "$build_log"
        else
            build_status="FAIL"
            BUILD_FAIL=$((BUILD_FAIL + 1))
            error_snippet=$(tail -5 "$build_log" | head -3)
        fi
    fi

    # --- TEST ---
    if [[ "$MODE" != "--build-only" ]]; then
        test_target=$(detect_test_target "$makefile")

        if [[ -z "$test_target" ]]; then
            test_status="SKIP"
            TEST_SKIP=$((TEST_SKIP + 1))
        else
            test_log="$FAIL_LOG_DIR/${rel_dir//\//_}_test.log"

            # Build the test target first (if separate from run)
            if make -C "$dir" "$test_target" >"$test_log" 2>&1; then
                # If the target builds a binary, try to find and run it
                # Check if there's a run_test target that actually executes
                if [[ "$test_target" == "test" ]] || [[ "$test_target" == "run_test" ]]; then
                    test_status="PASS"
                    TEST_PASS=$((TEST_PASS + 1))
                    rm -f "$test_log"
                else
                    # Custom test target — try running the built binary
                    if [[ -x "$dir/$test_target" ]]; then
                        if "$dir/$test_target" >>"$test_log" 2>&1; then
                            test_status="PASS"
                            TEST_PASS=$((TEST_PASS + 1))
                            rm -f "$test_log"
                        else
                            test_status="FAIL"
                            TEST_FAIL=$((TEST_FAIL + 1))
                            error_snippet=$(tail -10 "$test_log" | head -5)
                        fi
                    else
                        test_status="PASS"
                        TEST_PASS=$((TEST_PASS + 1))
                        rm -f "$test_log"
                    fi
                fi
            else
                test_status="FAIL"
                TEST_FAIL=$((TEST_FAIL + 1))
                error_snippet=$(tail -10 "$test_log" | head -5)
            fi
        fi
    fi

    # Print live status
    if [[ "$build_status" == "FAIL" ]] || [[ "$test_status" == "FAIL" ]]; then
        echo -e "${RED}FAIL${NC}  $rel_dir  (build=$build_status test=$test_status)"
    elif [[ "$test_status" == "SKIP" ]]; then
        echo -e "${YELLOW}SKIP${NC}  $rel_dir  (build=$build_status test=$test_status)"
    else
        echo -e "${GREEN}PASS${NC}  $rel_dir  (build=$build_status test=$test_status)"
    fi

    # Write report line
    echo -e "${rel_dir}\t${build_status}\t${test_status}\t${error_snippet}" >> "$REPORT_FILE"

done < <(find "$ROOT" -name Makefile -not -path '*/.git/*' -not -path '*/node_modules/*' | sort)

# --- SUMMARY ---
echo ""
echo -e "${CYAN}=== SUMMARY ===${NC}"
echo -e "Total projects:  $TOTAL"
echo -e "Build:  ${GREEN}$BUILD_PASS pass${NC}  ${RED}$BUILD_FAIL fail${NC}  $BUILD_SKIP skip"
echo -e "Test:   ${GREEN}$TEST_PASS pass${NC}  ${RED}$TEST_FAIL fail${NC}  ${YELLOW}$TEST_SKIP skip (no test target)${NC}"
echo ""

# List failures
if [[ $BUILD_FAIL -gt 0 ]] || [[ $TEST_FAIL -gt 0 ]]; then
    echo -e "${RED}=== FAILURES ===${NC}"
    grep -E 'FAIL' "$REPORT_FILE" | while IFS=$'\t' read -r dir bstat tstat err; do
        echo -e "${RED}$dir${NC}  build=$bstat  test=$tstat"
        if [[ -n "$err" ]]; then
            echo "  $err"
        fi
    done
    echo ""
    echo -e "Failure logs: $FAIL_LOG_DIR"
fi

echo -e "Full report: $REPORT_FILE"
