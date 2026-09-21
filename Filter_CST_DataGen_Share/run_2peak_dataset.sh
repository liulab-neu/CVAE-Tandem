#!/usr/bin/env bash
# Regenerate the 2peak CST training dataset (R x n_host1 x n_host2 sweep).
#
# Drives gen_2peak_dataset.py under xvfb.  With --jobs N the grid is split
# into N shards that run concurrently; each shard gets a private copy of the
# CST project (only Model/ + ModelCache/, about 4 MB) because every sample
# clears the project result folder, plus a disjoint slice of the CPU list.
#
# Every shard is restarted automatically after a crash.  Sample ids are a
# pure function of the grid and the script resumes from the exports already
# on disk, so a restart continues instead of redoing solved samples.
#
# Grid defaults match 1peak on the index axis: n = 1.5 .. 3.05 step 0.05.
# That is 41 x 32 x 32 = 41984 samples, so check --dry-run first.
#
# Usage examples
#   ./run_2peak_dataset.sh --dry-run                       # sizes and ids only
#   ./run_2peak_dataset.sh --jobs 4                        # full 3.05/0.05 grid
#   ./run_2peak_dataset.sh --jobs 4 --n-step 0.1           # 17 x 17 grid
#   ./run_2peak_dataset.sh --jobs 4 --end-id 200           # bounded pilot run
#   ./run_2peak_dataset.sh --status                        # progress only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MASTER_PROJECT="$SCRIPT_DIR/model2021_nk_model_multi_peaks.cst"

# Defaults ---------------------------------------------------------------
R_START=2.5
R_STOP=6.5
R_STEP=0.1
N_START=1.5
N_STOP=3.05
N_STEP=0.05
THETA=0
THETA_LIST=""
JOBS=1
START_ID=1
END_ID=0
ATTEMPTS=200
RETRY_DELAY=30
REOPEN_EVERY=1
RESTART_EVERY=300
STALL_MINUTES=30
OUTPUT_DIR=""
CPU_LIST="${TANDEM_CPU_LIST:-0-7,12-31}"
PYTHON_BIN="${TANDEM_PYTHON:-/home/yuxiao/miniconda3/envs/pytorch/bin/python}"
LOG_DIR="$REPO_ROOT/2peak/logs_dataset"
WORK_ROOT="$SCRIPT_DIR/parallel_work"
DRY_RUN=0
STATUS_ONLY=0
CLEAN_TEMP=0

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --r-start)      R_START="$2"; shift 2 ;;
        --r-stop)       R_STOP="$2"; shift 2 ;;
        --r-step)       R_STEP="$2"; shift 2 ;;
        --n-start)      N_START="$2"; shift 2 ;;
        --n-stop)       N_STOP="$2"; shift 2 ;;
        --n-step)       N_STEP="$2"; shift 2 ;;
        --theta)        THETA="$2"; shift 2 ;;
        --theta-list)   THETA_LIST="$2"; shift 2 ;;
        --jobs)         JOBS="$2"; shift 2 ;;
        --start-id)     START_ID="$2"; shift 2 ;;
        --end-id)       END_ID="$2"; shift 2 ;;
        --attempts)     ATTEMPTS="$2"; shift 2 ;;
        --retry-delay)  RETRY_DELAY="$2"; shift 2 ;;
        --reopen-every) REOPEN_EVERY="$2"; shift 2 ;;
        --restart-every) RESTART_EVERY="$2"; shift 2 ;;
        --stall-minutes) STALL_MINUTES="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --cpu-list)     CPU_LIST="$2"; shift 2 ;;
        --python)       PYTHON_BIN="$2"; shift 2 ;;
        --log-dir)      LOG_DIR="$2"; shift 2 ;;
        --work-root)    WORK_ROOT="$2"; shift 2 ;;
        --clean-temp)   CLEAN_TEMP=1; shift ;;
        --dry-run)      DRY_RUN=1; shift ;;
        --status)       STATUS_ONLY=1; shift ;;
        -h|--help)      usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

[[ -x "$PYTHON_BIN" ]] || { echo "Python interpreter not found: $PYTHON_BIN" >&2; exit 1; }
command -v xvfb-run >/dev/null || { echo "xvfb-run is required." >&2; exit 1; }
[[ -f "$MASTER_PROJECT" ]] || { echo "CST project not found: $MASTER_PROJECT" >&2; exit 1; }
[[ "$JOBS" =~ ^[0-9]+$ && "$JOBS" -ge 1 ]] || { echo "--jobs must be a positive integer." >&2; exit 1; }

GRID_ARGS=(
    --r-start "$R_START" --r-stop "$R_STOP" --r-step "$R_STEP"
    --n-start "$N_START" --n-stop "$N_STOP" --n-step "$N_STEP"
    --theta "$THETA"
    --start-id "$START_ID" --end-id "$END_ID"
    --reopen-every "$REOPEN_EVERY"
    --max-samples "$RESTART_EVERY"
)
[[ -n "$THETA_LIST" ]] && GRID_ARGS+=(--theta-list "$THETA_LIST")
[[ -n "$OUTPUT_DIR" ]] && GRID_ARGS+=(--output-dir "$OUTPUT_DIR")

# Plain Python start-up segfaults intermittently on this machine, so even the
# grid-only calls are pinned and retried.
run_generator_python() {
    local attempt
    for attempt in 1 2 3 4 5; do
        if taskset -c "$CPU_LIST" "$PYTHON_BIN" -u \
            "$SCRIPT_DIR/gen_2peak_dataset.py" "$@"; then
            return 0
        fi
        sleep 2
    done
    echo "gen_2peak_dataset.py failed 5 times: $*" >&2
    return 1
}

if [[ "$DRY_RUN" -eq 1 || "$STATUS_ONLY" -eq 1 ]]; then
    for ((shard = 0; shard < JOBS; shard++)); do
        echo "----- shard $((shard + 1))/$JOBS -----"
        run_generator_python \
            "${GRID_ARGS[@]}" --shard "$shard" --num-shards "$JOBS" --dry-run
    done
    exit 0
fi

# Expand "0-7,12-31" into an array of individual CPU ids. ----------------
expand_cpu_list() {
    local spec="$1" chunk start end cpu
    local -a cpus=()
    IFS=',' read -ra parts <<<"$spec"
    for chunk in "${parts[@]}"; do
        if [[ "$chunk" == *-* ]]; then
            start="${chunk%%-*}"; end="${chunk##*-}"
            for ((cpu = start; cpu <= end; cpu++)); do cpus+=("$cpu"); done
        else
            cpus+=("$chunk")
        fi
    done
    printf '%s\n' "${cpus[@]}"
}

mapfile -t ALL_CPUS < <(expand_cpu_list "$CPU_LIST")
TOTAL_CPUS=${#ALL_CPUS[@]}
if [[ "$JOBS" -gt "$TOTAL_CPUS" ]]; then
    echo "--jobs ($JOBS) exceeds the ${TOTAL_CPUS} CPUs in --cpu-list." >&2
    exit 1
fi

worker_cpu_list() {
    local index="$1"
    local base=$((TOTAL_CPUS / JOBS))
    local extra=$((TOTAL_CPUS % JOBS))
    local start=$((index * base + (index < extra ? index : extra)))
    local count=$((base + (index < extra ? 1 : 0)))
    local joined="" cpu
    for ((cpu = start; cpu < start + count; cpu++)); do
        joined+="${joined:+,}${ALL_CPUS[$cpu]}"
    done
    printf '%s' "$joined"
}

prepare_worker_project() {
    local index="$1"
    if [[ "$JOBS" -eq 1 ]]; then
        printf '%s' "$MASTER_PROJECT"
        return
    fi
    local base_name project_dir worker_dir target sub
    base_name="$(basename "${MASTER_PROJECT%.cst}")"
    project_dir="${MASTER_PROJECT%.cst}"
    worker_dir="$WORK_ROOT/ds$index"
    target="$worker_dir/$base_name.cst"

    rm -rf "$worker_dir"
    mkdir -p "$worker_dir/$base_name"
    cp "$MASTER_PROJECT" "$target"
    # Result/ and Temp/ are solver output and are the only large parts of the
    # project, so they are deliberately not copied.
    for sub in Model ModelCache Export SP; do
        [[ -d "$project_dir/$sub" ]] && cp -r "$project_dir/$sub" "$worker_dir/$base_name/"
    done
    printf '%s' "$target"
}

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "Repository root : $REPO_ROOT"
echo "Python          : $PYTHON_BIN"
echo "CPU affinity    : $CPU_LIST ($TOTAL_CPUS cpus)"
echo "Shards          : $JOBS"
echo "R axis          : $R_START .. $R_STOP step $R_STEP"
echo "n axis          : $N_START .. $N_STOP step $N_STEP"
echo "theta           : ${THETA_LIST:-$THETA} deg"
echo "Restarts/shard  : $ATTEMPTS crash retries; clean restart every $RESTART_EVERY samples"
echo "Stall watchdog  : kill a shard whose log is idle for $STALL_MINUTES min"
echo "Logs            : $LOG_DIR"
echo

if [[ "$CLEAN_TEMP" -eq 1 ]]; then
    temp_dir="${MASTER_PROJECT%.cst}/Temp"
    if [[ -d "$temp_dir" ]]; then
        echo "Clearing stale solver temp: $temp_dir ($(du -sh "$temp_dir" | cut -f1))"
        rm -rf "${temp_dir:?}"/*
    fi
fi

# Kill a process and every descendant. The shard tree is
# setsid -> taskset -> xvfb-run -> (Xvfb, python -> CST), and $! is only the
# outermost pid, so signalling that alone leaves the solver running.
kill_tree() {
    local root="$1" signal="${2:-KILL}" child
    for child in $(pgrep -P "$root" 2>/dev/null); do
        kill_tree "$child" "$signal"
    done
    kill "-$signal" "$root" 2>/dev/null || true
}

# One shard: restart the generator until it exits cleanly. ---------------
run_shard() {
    local shard="$1" project_file="$2" cpus="$3"
    local log_file="$LOG_DIR/shard${shard}_${STAMP}.log"
    local attempt

    local rc restarts=0
    for ((attempt = 1; attempt <= ATTEMPTS; attempt++)); do
        echo "[shard $shard] start attempt $attempt/$ATTEMPTS on cpus $cpus"
        rc=0
        taskset -c "$cpus" \
            env TANDEM_PYTHON="$PYTHON_BIN" MPLBACKEND=Agg \
            xvfb-run -a "$PYTHON_BIN" -u "$SCRIPT_DIR/gen_2peak_dataset.py" \
                "${GRID_ARGS[@]}" \
                --shard "$shard" --num-shards "$JOBS" \
                --cst-file "$project_file" \
            >>"$log_file" 2>&1 &
        local child=$!
        local stalled=0
        # CST can abort internally and leave the client waiting on a dead
        # design environment: the process never exits, so the crash-retry
        # path above never fires. Watch the log instead of the exit status.
        while kill -0 "$child" 2>/dev/null; do
            sleep 60
            kill -0 "$child" 2>/dev/null || break
            local idle=$(( ($(date +%s) - $(stat -c %Y "$log_file" 2>/dev/null || date +%s)) / 60 ))
            if (( idle >= STALL_MINUTES )); then
                echo "[shard $shard] stalled: log idle ${idle} min, killing the shard tree"
                tail -3 "$log_file" | sed 's/^/    /'
                kill_tree "$child" TERM
                sleep 5
                kill_tree "$child" KILL
                stalled=1
                break
            fi
        done
        wait "$child" 2>/dev/null || rc=$?
        (( stalled )) && rc=124
        if [[ $rc -eq 0 ]]; then
            echo "[shard $shard] finished"
            return 0
        fi
        if [[ $rc -eq 75 ]]; then
            # Planned memory-bounded exit, not a failure: restart immediately
            # and do not spend one of the crash retries on it.
            restarts=$((restarts + 1))
            echo "[shard $shard] clean restart #$restarts (memory bound)"
            attempt=$((attempt - 1))
            continue
        fi
        # A grid or argument mistake will never succeed on a retry; the CST
        # and Python start-up crashes on this machine will.
        if grep -qE '^(usage:|ValueError:|FileNotFoundError:)' "$log_file"; then
            echo "[shard $shard] configuration error, not restarting:"
            grep -E '^(usage:|ValueError:|FileNotFoundError:)' "$log_file" | tail -3 | sed 's/^/    /'
            return 1
        fi
        echo "[shard $shard] crashed, last log lines:"
        tail -4 "$log_file" | sed 's/^/    /'
        (( attempt < ATTEMPTS )) && sleep "$RETRY_DELAY"
    done
    echo "[shard $shard] gave up after $ATTEMPTS attempts (see $log_file)"
    return 1
}

PIDS=()
for ((shard = 0; shard < JOBS; shard++)); do
    cpus="$(worker_cpu_list "$shard")"
    project="$(prepare_worker_project "$shard")"
    echo "shard $shard: cpus=$cpus project=$project"
    if [[ "$JOBS" -eq 1 ]]; then
        run_shard "$shard" "$project" "$cpus" || true
    else
        run_shard "$shard" "$project" "$cpus" &
        PIDS+=("$!")
    fi
done
echo

for pid in ${PIDS[@]+"${PIDS[@]}"}; do
    wait "$pid" || true
done

echo "==================== summary ===================="
run_generator_python "${GRID_ARGS[@]}" --shard 0 --num-shards 1 --dry-run | tail -5
