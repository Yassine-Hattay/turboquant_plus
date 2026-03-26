#!/bin/bash
# TurboQuant Hardware Diagnostic v3
# Comprehensive benchmark + device profiling + load monitoring.
# Run on ANY hardware. Send the output file back for analysis.
#
# NO PII collected — only hardware specs, load stats, and benchmark numbers.
#
# Usage:
#   bash turbo-hardware-diag.sh [path-to-llama-cpp] [path-to-model.gguf]
#
# Output: turbo-diag-<date>.json  (machine-parseable)
#         turbo-diag-<date>.txt   (human-readable log)
#
# Tests:
#   1. Hardware inventory (GPU, memory, thermals, no PII)
#   2. System load snapshot (memory pressure, swap, GPU utilization)
#   3. Metal device capabilities (GPU family, tensor API, simdgroup features)
#   4. Build validation (turbo3 type available, Metal library loads)
#   5. Prefill speed at 2K/4K/8K/16K/32K for q8_0, turbo3, mode 2
#   6. Decode speed at short/4K/8K/16K/32K depth for q8_0, turbo3, mode 2
#   7. Constant cache stress test (fine-grained decode scaling gradient)
#   8. Combined prefill+decode (realistic workload)
#   9. PPL validation (quality gate)
#  10. Memory breakdown (KV cache sizing at multiple context lengths)
#  11. Load monitoring during benchmarks (detect thermal throttling)
#
# Estimated runtime: 20-40 minutes depending on hardware.

set -uo pipefail

# --- Args ---
LLAMA_DIR="${1:-$(pwd)}"
MODEL="${2:-}"
WIKI="${LLAMA_DIR}/wikitext-2-raw/wiki.test.raw"
BENCH="${LLAMA_DIR}/build/bin/llama-bench"
PERPL="${LLAMA_DIR}/build/bin/llama-perplexity"
CLI="${LLAMA_DIR}/build/bin/llama-cli"

# Auto-find model if not specified
if [ -z "$MODEL" ]; then
    MODEL=$(find "${LLAMA_DIR}/models" "${LLAMA_DIR}/../models" /Users/*/local_llms/models -name "*.gguf" -type f 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then
        echo "ERROR: No .gguf model found. Pass model path as second argument."
        echo "Usage: bash turbo-hardware-diag.sh /path/to/llama.cpp /path/to/model.gguf"
        exit 1
    fi
fi

DATE=$(date +%Y%m%d-%H%M%S)
OUTFILE="turbo-diag-${DATE}.txt"
JSONFILE="turbo-diag-${DATE}.json"

# --- Validate tools exist ---
for tool in "$BENCH" "$PERPL" "$CLI"; do
    if [ ! -f "$tool" ]; then
        echo "ERROR: $tool not found. Build llama.cpp first:"
        echo "  cd $LLAMA_DIR"
        echo "  cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_BUILD_TYPE=Release"
        echo "  cmake --build build -j"
        exit 1
    fi
done

# --- Helpers ---
section() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

subsection() {
    echo ""
    echo "--- $1 ---"
}

# Capture system load snapshot (no PII)
capture_load() {
    local label="$1"
    echo ""
    echo "[LOAD_SNAPSHOT] label=$label timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    # Load average (cross-platform)
    if [ -f /proc/loadavg ]; then
        echo "[LOAD_SNAPSHOT] load_avg=$(cat /proc/loadavg | awk '{print $1, $2, $3}')"
    elif command -v sysctl &>/dev/null; then
        echo "[LOAD_SNAPSHOT] load_avg=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{print $1, $2, $3}' || echo 'unknown')"
    fi

    # Memory info
    if [ "$(uname -s)" = "Darwin" ]; then
        # macOS: vm_stat + swap
        if command -v vm_stat &>/dev/null; then
            local pageins pageouts swapused free_mem
            pageins=$(vm_stat 2>/dev/null | grep "Pageins" | awk '{print $2}' | tr -d '.' || echo "0")
            pageouts=$(vm_stat 2>/dev/null | grep "Pageouts" | awk '{print $2}' | tr -d '.' || echo "0")
            swapused=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' || echo "unknown")
            free_mem=$(vm_stat 2>/dev/null | awk '/Pages free/ {free=$3} /Pages inactive/ {inactive=$3} END {gsub(/\./,"",free); gsub(/\./,"",inactive); printf "%.0f MB", (free+inactive)*4096/1048576}' || echo "unknown")
            echo "[LOAD_SNAPSHOT] page_ins=$pageins page_outs=$pageouts swap_used=$swapused"
            echo "[LOAD_SNAPSHOT] approx_free_ram=$free_mem"
        fi
        if command -v memory_pressure &>/dev/null; then
            local mp
            mp=$(memory_pressure 2>/dev/null | grep "System-wide" | head -1 || echo "unknown")
            echo "[LOAD_SNAPSHOT] memory_pressure=$mp"
        fi
        # Thermal
        if command -v pmset &>/dev/null; then
            echo "[LOAD_SNAPSHOT] thermal=$(pmset -g therm 2>/dev/null | grep -i 'cpu_speed_limit' || echo 'no thermal data')"
        fi
        # GPU utilization
        if command -v ioreg &>/dev/null; then
            echo "[LOAD_SNAPSHOT] gpu_ioreg=$(ioreg -r -d 1 -c IOAccelerator 2>/dev/null | grep -i 'PerformanceState' | head -1 || echo 'no GPU metrics')"
        fi
    elif [ "$(uname -s)" = "Linux" ]; then
        # Linux: /proc/meminfo + swap
        if [ -f /proc/meminfo ]; then
            local mem_free mem_avail swap_total swap_free
            mem_free=$(grep MemFree /proc/meminfo | awk '{print $2}')
            mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
            swap_total=$(grep SwapTotal /proc/meminfo | awk '{print $2}')
            swap_free=$(grep SwapFree /proc/meminfo | awk '{print $2}')
            echo "[LOAD_SNAPSHOT] mem_free_mb=$((mem_free / 1024)) mem_available_mb=$((mem_avail / 1024))"
            echo "[LOAD_SNAPSHOT] swap_total_mb=$((swap_total / 1024)) swap_free_mb=$((swap_free / 1024))"
        fi
        # GPU utilization (NVIDIA)
        if command -v nvidia-smi &>/dev/null; then
            echo "[LOAD_SNAPSHOT] gpu_util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
        fi
        # Thermal
        if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
            echo "[LOAD_SNAPSHOT] cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)m°C"
        fi
    fi

    # Process count (cross-platform)
    echo "[LOAD_SNAPSHOT] process_count=$(ps aux 2>/dev/null | wc -l | tr -d ' ' || echo 'unknown')"
}

run_bench() {
    local label="$1"
    local ctk="$2"
    local ctv="$3"
    local extra_args="$4"
    local env_prefix="${5:-}"

    subsection "$label"
    echo "[BENCH_START] label=\"$label\" ctk=$ctk ctv=$ctv args=\"$extra_args\" env=\"$env_prefix\" timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local start_ms
    start_ms=$(date +%s%3N 2>/dev/null || date +%s)
    eval "${env_prefix} ${BENCH} -m \"${MODEL}\" -ngl 99 -fa 1 -ctk ${ctk} -ctv ${ctv} ${extra_args} -r 3 2>&1" || echo "FAILED: $label"
    local end_ms
    end_ms=$(date +%s%3N 2>/dev/null || date +%s)
    echo "[BENCH_END] label=\"$label\" wall_ms=$((end_ms - start_ms))"
}

run_perpl() {
    local label="$1"
    local ctk="$2"
    local ctv="$3"
    local chunks="$4"
    local env_prefix="${5:-}"

    subsection "$label"
    echo "[PPL_START] label=\"$label\" ctk=$ctk ctv=$ctv chunks=$chunks timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    eval "${env_prefix} ${PERPL} -m \"${MODEL}\" -ngl 99 -fa on --cache-type-k ${ctk} --cache-type-v ${ctv} -f \"${WIKI}\" --chunks ${chunks} 2>&1" || echo "FAILED: $label"
    echo "[PPL_END] label=\"$label\""
}

# ===================================================================
echo "TurboQuant Hardware Diagnostic v3"
echo "Output: $OUTFILE (human-readable) + $JSONFILE (machine-parseable)"
echo "Model: $MODEL"
echo ""
echo "NO PII is collected. Only hardware specs, load stats, and benchmarks."
echo "Estimated runtime: 20-40 minutes."
echo ""
echo "Press Ctrl+C to abort at any time."
echo ""

# --- Start capture ---
exec > >(tee "$OUTFILE") 2>&1

echo "TURBO_DIAG_VERSION=3"
echo "TURBO_DIAG_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "TURBO_DIAG_MODEL=$(basename "$MODEL")"
echo "TURBO_DIAG_MODEL_SIZE=$(ls -lh "$MODEL" 2>/dev/null | awk '{print $5}' || echo "unknown")"

# ===================================================================
section "1. HARDWARE INVENTORY (no PII)"
# ===================================================================

PLATFORM="$(uname -s)"
echo "[HW] os=$PLATFORM os_version=$(uname -r) arch=$(uname -m)"

if [ "$PLATFORM" = "Darwin" ]; then
    # ---- macOS ----
    echo "[HW] kernel=$(sw_vers -productVersion 2>/dev/null || uname -r)"
    echo "[HW] cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
    echo "[HW] cpu_cores_physical=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 'unknown')"
    echo "[HW] cpu_cores_logical=$(sysctl -n hw.logicalcpu 2>/dev/null || echo 'unknown')"
    echo "[HW] cpu_freq_max=$(sysctl -n hw.cpufrequency_max 2>/dev/null | awk '{printf "%.0f MHz", $1/1000000}' 2>/dev/null || echo 'unknown')"
    echo "[HW] ram_total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo '0')"
    echo "[HW] ram_total_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}' || echo 'unknown')"

    # GPU from system_profiler
    subsection "GPU Details"
    if command -v system_profiler &>/dev/null; then
        system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Total Number|Metal|Cores|Model" | sed 's/^[[:space:]]*/[HW_GPU] /' || true
    fi

    # Apple Silicon detection
    chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "")
    if echo "$chip" | grep -q "Apple"; then
        echo "[HW] apple_silicon=true"
        echo "[HW] chip_model=$chip"
    else
        echo "[HW] apple_silicon=false"
    fi

    # Cache hierarchy
    echo "[HW] page_size=$(sysctl -n hw.pagesize 2>/dev/null || echo 'unknown')"
    echo "[HW] l1_dcache=$(sysctl -n hw.l1dcachesize 2>/dev/null || echo 'unknown')"
    echo "[HW] l2_cache=$(sysctl -n hw.l2cachesize 2>/dev/null || echo 'unknown')"

    # Thermal / power state
    if command -v pmset &>/dev/null; then
        subsection "Power State"
        pmset -g batt 2>/dev/null | grep -E "Now drawing|charging|AC Power|Battery" | sed 's/^/[HW_POWER] /' || true
        echo "[HW_POWER] cpu_speed_limit=$(pmset -g therm 2>/dev/null | grep "CPU_Speed_Limit" | awk '{print $3}' || echo 'unknown')"
    fi

elif [ "$PLATFORM" = "Linux" ]; then
    # ---- Linux ----
    echo "[HW] kernel=$(uname -r)"
    echo "[HW] cpu_brand=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo 'unknown')"
    echo "[HW] cpu_cores_physical=$(grep -c 'processor' /proc/cpuinfo 2>/dev/null || echo 'unknown')"
    echo "[HW] cpu_cores_logical=$(nproc 2>/dev/null || echo 'unknown')"
    ram_kb=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}' || echo "0")
    echo "[HW] ram_total_bytes=$((ram_kb * 1024))"
    echo "[HW] ram_total_gb=$((ram_kb / 1048576))"

    # GPU detection (NVIDIA / AMD)
    subsection "GPU Details"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version,gpu_bus_id --format=csv,noheader 2>/dev/null | sed 's/^/[HW_GPU] /' || true
        echo "[HW] gpu_backend=cuda"
    elif [ -d /sys/class/drm ]; then
        for card in /sys/class/drm/card*/device; do
            if [ -f "$card/vendor" ]; then
                echo "[HW_GPU] pci_vendor=$(cat "$card/vendor" 2>/dev/null) pci_device=$(cat "$card/device" 2>/dev/null)"
            fi
        done
        echo "[HW] gpu_backend=vulkan_or_other"
    else
        echo "[HW_GPU] no GPU detected"
    fi

    echo "[HW] apple_silicon=false"

    # Cache hierarchy
    for level in 1 2 3; do
        size=$(cat /sys/devices/system/cpu/cpu0/cache/index${level}/size 2>/dev/null || echo "unknown")
        echo "[HW] l${level}_cache=$size"
    done

    # Thermal
    subsection "Thermal"
    if [ -d /sys/class/thermal ]; then
        for tz in /sys/class/thermal/thermal_zone*/temp; do
            temp=$(cat "$tz" 2>/dev/null || echo "0")
            zone=$(basename "$(dirname "$tz")")
            echo "[HW_THERMAL] $zone=${temp}m°C"
        done
    fi

    # Power (laptop vs plugged in)
    if [ -d /sys/class/power_supply ]; then
        for ps_dir in /sys/class/power_supply/*/; do
            ptype=$(cat "${ps_dir}type" 2>/dev/null || echo "unknown")
            status=$(cat "${ps_dir}status" 2>/dev/null || echo "unknown")
            echo "[HW_POWER] $(basename "$ps_dir") type=$ptype status=$status"
        done
    fi
fi

# ===================================================================
section "2. SYSTEM LOAD (pre-benchmark baseline)"
# ===================================================================

echo "Capturing system state BEFORE benchmarks to detect interference."
capture_load "pre_benchmark"

# Top processes by CPU (command names only, no args/paths = no PII)
subsection "Top CPU consumers (command name only)"
ps -eo pcpu,comm 2>/dev/null | sort -rn | head -10 | sed 's/^/[LOAD_TOP] /' || true

# Disk I/O state
subsection "Disk I/O"
if command -v iostat &>/dev/null; then
    iostat -c 1 2>/dev/null | head -5 | sed 's/^/[LOAD_IO] /' || true
fi

# ===================================================================
section "3. MODEL INFO"
# ===================================================================

echo "Extracting model metadata from GGUF file..."
echo ""

# Run a quick CLI init to capture model metadata
"$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 512 -n 0 -p "x" --jinja 2>&1 | grep -E "general\.name|general\.architecture|general\.file_type|general\.size_label|general\.quantized_by|general\.base_model|model type|model params|file type|file format|file size|n_ctx_train|n_embd|n_layer|n_head|n_head_kv|n_expert|n_expert_used|n_embd_head|vocab type|n_vocab|arch" | sed 's/^/[MODEL] /' || true

echo ""
echo "[MODEL] filename=$(basename "$MODEL")"
echo "[MODEL] filesize=$(ls -lh "$MODEL" 2>/dev/null | awk '{print $5}' || echo 'unknown')"
echo "[MODEL] filesize_bytes=$(stat -f%z "$MODEL" 2>/dev/null || stat --printf="%s" "$MODEL" 2>/dev/null || echo '0')"

# ===================================================================
section "4. GPU DEVICE CAPABILITIES"
# ===================================================================

echo "Extracting GPU info from llama.cpp init..."
echo ""

# Run a minimal CLI invocation to get device init logs
GPU_INIT=$("$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k turbo3 --cache-type-v turbo3 -c 512 -n 1 -p "test" --jinja 2>&1)
echo "$GPU_INIT" | grep -E "build:|GPU name|GPU family|simdgroup|unified|bfloat|has tensor|residency|shared buffers|recommendedMax|system.info|system_info|n_threads|turbo|TurboQuant|KV buffer|rotation|metal_library|metal_init|embed|loaded in|CUDA|cuda|VRAM|cublas" | sed 's/^/[GPU] /' || true

if [ "$PLATFORM" = "Darwin" ]; then
    # macOS: check for Metal tensor API (critical for M1 decode perf)
    subsection "Tensor API Check (M1 vs M5 decode performance)"
    has_tensor_line=$(echo "$GPU_INIT" | grep "has tensor" || echo "NOT FOUND")
    echo "[METAL_TENSOR] $has_tensor_line"
    if echo "$has_tensor_line" | grep -q "false"; then
        echo "[METAL_TENSOR] WARNING: Tensor API disabled. This is M1/M2/M3/M4 hardware."
        echo "[METAL_TENSOR] WARNING: Turbo3 decode may be significantly slower due to constant cache limitations."
    fi
elif [ "$PLATFORM" = "Linux" ]; then
    subsection "CUDA Device Check"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm --format=csv,noheader 2>/dev/null | sed 's/^/[CUDA] /' || true
    else
        echo "[CUDA] nvidia-smi not found — CUDA may not be available"
    fi
fi

# ===================================================================
section "5. BUILD VALIDATION"
# ===================================================================

# Verify turbo3 works in llama-bench
subsection "turbo3 in llama-bench"
"$BENCH" -m "$MODEL" -ngl 99 -fa 1 -ctk turbo3 -ctv turbo3 -p 64 -n 0 -r 1 2>&1 | tail -5 || echo "FAILED: turbo3 not available in llama-bench"

# Verify Metal library loaded (not silent CPU fallback)
subsection "Metal library validation"
"$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k turbo3 --cache-type-v turbo3 -c 512 -n 1 -p "test" --jinja 2>&1 | grep -E "metal_library|embed|loaded in" | head -5 || echo "WARNING: Could not verify Metal library load"

# Git commit hash
subsection "Build commit"
(cd "$LLAMA_DIR" && git log --oneline -1 2>/dev/null || echo "not a git repo") | sed 's/^/[BUILD] /'

# ===================================================================
section "6. PREFILL SPEED (tok/s)"
# ===================================================================

echo "Prefill at 2K/4K/8K/16K/32K. Expect flat 0.98-1.00x q8_0."
echo "If ratio drops >5% across depths = context scaling regression."
echo ""

capture_load "pre_prefill"

run_bench "q8_0 prefill (all depths)" "q8_0" "q8_0" "-p 2048,4096,8192,16384,32768 -n 0"
run_bench "turbo3 prefill (all depths)" "turbo3" "turbo3" "-p 2048,4096,8192,16384,32768 -n 0"
run_bench "turbo3 mode2 prefill (all depths)" "turbo3" "turbo3" "-p 2048,4096,8192,16384,32768 -n 0" "TURBO_LAYER_ADAPTIVE=2"

capture_load "post_prefill"

# ===================================================================
section "7. DECODE SPEED (tok/s) — THE CRITICAL TEST"
# ===================================================================

echo "Decode at increasing context depths. This is where M1 fails."
echo ""
echo "Known baselines:"
echo "  M5 Max: turbo3/q8_0 = 0.92x (short) → 0.72x (48K)"
echo "  M1 Max: turbo3/q8_0 = ??? (short) → 0.09x (42K) ← CATASTROPHIC"
echo ""
echo "Healthy: ratio stays above 0.70x through 32K"
echo "Problem: ratio drops below 0.50x at any depth"
echo ""

capture_load "pre_decode"

# Short context (baseline — no KV pressure)
run_bench "q8_0 decode (short)" "q8_0" "q8_0" "-p 0 -n 128"
run_bench "turbo3 decode (short)" "turbo3" "turbo3" "-p 0 -n 128"
run_bench "turbo3 mode2 decode (short)" "turbo3" "turbo3" "-p 0 -n 128" "TURBO_LAYER_ADAPTIVE=2"

# 4K depth
run_bench "q8_0 decode @4K" "q8_0" "q8_0" "-p 0 -n 128 -d 4096"
run_bench "turbo3 decode @4K" "turbo3" "turbo3" "-p 0 -n 128 -d 4096"
run_bench "turbo3 mode2 decode @4K" "turbo3" "turbo3" "-p 0 -n 128 -d 4096" "TURBO_LAYER_ADAPTIVE=2"

# 8K depth
run_bench "q8_0 decode @8K" "q8_0" "q8_0" "-p 0 -n 128 -d 8192"
run_bench "turbo3 decode @8K" "turbo3" "turbo3" "-p 0 -n 128 -d 8192"
run_bench "turbo3 mode2 decode @8K" "turbo3" "turbo3" "-p 0 -n 128 -d 8192" "TURBO_LAYER_ADAPTIVE=2"

# 16K depth
run_bench "q8_0 decode @16K" "q8_0" "q8_0" "-p 0 -n 128 -d 16384"
run_bench "turbo3 decode @16K" "turbo3" "turbo3" "-p 0 -n 128 -d 16384"
run_bench "turbo3 mode2 decode @16K" "turbo3" "turbo3" "-p 0 -n 128 -d 16384" "TURBO_LAYER_ADAPTIVE=2"

# 32K depth
run_bench "q8_0 decode @32K" "q8_0" "q8_0" "-p 0 -n 128 -d 32768"
run_bench "turbo3 decode @32K" "turbo3" "turbo3" "-p 0 -n 128 -d 32768"
run_bench "turbo3 mode2 decode @32K" "turbo3" "turbo3" "-p 0 -n 128 -d 32768" "TURBO_LAYER_ADAPTIVE=2"

capture_load "post_decode"

# ===================================================================
section "8. CONSTANT CACHE STRESS TEST (fine-grained decode gradient)"
# ===================================================================

echo "Fine-grained decode at many depths to find the EXACT inflection point."
echo "This tells us where constant cache pressure becomes dominant."
echo ""
echo "NOTE: 1K context is unreliable (Metal async dispatch timing artifact)."
echo "      Results at 1K may show impossibly high numbers — ignore them."
echo ""

for DEPTH in 2048 3072 4096 6144 8192 12288 16384 20480 24576 28672 32768; do
    run_bench "turbo3 decode @${DEPTH} (stress)" "turbo3" "turbo3" "-p 0 -n 64 -d $DEPTH"
done

echo ""
echo "q8_0 baseline at same depths:"
for DEPTH in 2048 3072 4096 6144 8192 12288 16384 20480 24576 28672 32768; do
    run_bench "q8_0 decode @${DEPTH} (stress)" "q8_0" "q8_0" "-p 0 -n 64 -d $DEPTH"
done

capture_load "post_stress"

# ===================================================================
section "9. COMBINED PREFILL+DECODE (realistic workload)"
# ===================================================================

echo "Simulates real usage: prefill a prompt, then generate response tokens."
echo ""

run_bench "q8_0 pp4K+tg128" "q8_0" "q8_0" "-pg 4096,128"
run_bench "turbo3 pp4K+tg128" "turbo3" "turbo3" "-pg 4096,128"
run_bench "turbo3 mode2 pp4K+tg128" "turbo3" "turbo3" "-pg 4096,128" "TURBO_LAYER_ADAPTIVE=2"

run_bench "q8_0 pp8K+tg256" "q8_0" "q8_0" "-pg 8192,256"
run_bench "turbo3 pp8K+tg256" "turbo3" "turbo3" "-pg 8192,256"
run_bench "turbo3 mode2 pp8K+tg256" "turbo3" "turbo3" "-pg 8192,256" "TURBO_LAYER_ADAPTIVE=2"

run_bench "q8_0 pp16K+tg512" "q8_0" "q8_0" "-pg 16384,512"
run_bench "turbo3 pp16K+tg512" "turbo3" "turbo3" "-pg 16384,512"

# ===================================================================
section "10. PERPLEXITY (quality validation)"
# ===================================================================

echo "PPL must be within 2% of q8_0. If not, something is wrong with the build."
echo "CRITICAL: A PPL >2% delta means quality is broken — speed numbers are meaningless."
echo ""

if [ -f "$WIKI" ]; then
    run_perpl "q8_0 PPL (8 chunks)" "q8_0" "q8_0" "8"
    run_perpl "turbo3 PPL (8 chunks)" "turbo3" "turbo3" "8"
    run_perpl "turbo3 mode2 PPL (8 chunks)" "turbo3" "turbo3" "8" "TURBO_LAYER_ADAPTIVE=2"
else
    echo "SKIPPED: wikitext-2-raw not found at $WIKI"
    echo ""
    echo "To enable PPL testing, download wikitext-2:"
    echo "  wget https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1.zip"
    echo "  unzip wikitext-2-raw-v1.zip -d ${LLAMA_DIR}/wikitext-2-raw"
    echo ""
    echo "PPL is the single most important quality check. Without it, speed numbers"
    echo "could be from a broken build that outputs garbage at high speed."
fi

# ===================================================================
section "11. MEMORY BREAKDOWN"
# ===================================================================

subsection "KV cache at 32K context"
"$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k turbo3 --cache-type-v turbo3 -c 32768 -n 1 -p "test" --jinja 2>&1 | grep -E "KV buffer|KV.*size|memory_breakdown|compute buffer|RS buffer|model buffer|recommendedMax|load_tensors|offload" | sed 's/^/[MEM_32K] /' || true

subsection "KV cache at 131K context"
"$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k turbo3 --cache-type-v turbo3 -c 131072 -n 1 -p "test" --jinja 2>&1 | grep -E "KV buffer|KV.*size|memory_breakdown|compute buffer|RS buffer|model buffer|recommendedMax|load_tensors|offload" | sed 's/^/[MEM_131K] /' || true

subsection "q8_0 KV at 32K (for size comparison)"
"$CLI" -m "$MODEL" -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -c 32768 -n 1 -p "test" --jinja 2>&1 | grep -E "KV buffer|KV.*size" | sed 's/^/[MEM_Q8_32K] /' || true

# ===================================================================
section "12. SYSTEM LOAD (post-benchmark)"
# ===================================================================

echo "Final load snapshot — compare with pre-benchmark to detect thermal throttling."
capture_load "post_all_benchmarks"

# Check if thermal throttling may have occurred
subsection "Thermal throttling check"
if [ "$PLATFORM" = "Darwin" ] && command -v pmset &>/dev/null; then
    final_limit=$(pmset -g therm 2>/dev/null | grep "CPU_Speed_Limit" | awk '{print $3}' || echo "100")
    echo "[THERMAL] final_cpu_speed_limit=$final_limit"
    if [ "${final_limit:-100}" -lt 100 ] 2>/dev/null; then
        echo "[THERMAL] WARNING: CPU speed limited to ${final_limit}%. Results may be throttled."
        echo "[THERMAL] WARNING: Run benchmarks again after cooling down for accurate results."
    fi
elif [ "$PLATFORM" = "Linux" ] && [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    final_temp=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "0")
    echo "[THERMAL] final_cpu_temp=${final_temp}m°C"
    if [ "${final_temp:-0}" -gt 90000 ] 2>/dev/null; then
        echo "[THERMAL] WARNING: CPU temperature above 90°C. Results may be thermally throttled."
    fi
fi

# ===================================================================
section "13. DIAGNOSTIC SUMMARY"
# ===================================================================

echo "TURBO_DIAG_COMPLETE=true"
echo "TURBO_DIAG_END_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "Output saved to:"
echo "  $OUTFILE (human-readable, this file)"
echo ""
echo "============================================"
echo "  HOW TO READ THESE RESULTS"
echo "============================================"
echo ""
echo "1. MODEL: Section 3 confirms which model and quantization you're running."
echo ""
echo "2. QUALITY: Section 10 PPL should be within 2% of q8_0."
echo "   If PPL is broken, ALL speed numbers are meaningless."
echo ""
echo "3. PREFILL: Section 6 turbo3/q8_0 ratio should be flat 0.95-1.00x"
echo "   across all depths. If it degrades = context scaling regression."
echo ""
echo "4. DECODE: Section 7 is the critical metric."
echo "   Healthy (M5):  0.92x (short) -> 0.72x (48K) — gradual degradation"
echo "   Problem (M1):  0.90x (short) -> 0.09x (42K) — constant cache thrashing"
echo ""
echo "5. INFLECTION POINT: Section 8 stress test shows the exact depth"
echo "   where turbo3 decode starts falling off a cliff."
echo ""
echo "6. THERMAL: Compare Sections 2 and 12. If CPU_Speed_Limit dropped"
echo "   during the test, results may be artificially low."
echo ""
echo "7. MEMORY: Section 11 shows KV cache sizing. If the system is near"
echo "   its recommendedMaxWorkingSetSize, swap pressure kills decode."
echo ""
echo "NEXT STEPS:"
echo "  Send this file to the TurboQuant team for analysis."
echo "  If decode ratio < 0.50x at any depth, use TURBO_LAYER_ADAPTIVE=2"
echo "  or q8_0 cache until the M1 constant cache fix is available."
echo ""
echo "END OF DIAGNOSTIC"

# ===================================================================
# PACKAGE RESULTS
# ===================================================================

# Stop tee capture before packaging
exec 1>&1 2>&2

echo ""
echo "Packaging results..."

ZIPFILE="turbo-diag-${DATE}.zip"

# Collect all diagnostic artifacts
zip -j "$ZIPFILE" "$OUTFILE" 2>/dev/null

# Add a machine-readable hardware profile for replay testing
PROFILE="turbo-hwprofile-${DATE}.json"
cat > "$PROFILE" << HWEOF
{
  "diag_version": 3,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platform": "$PLATFORM",
  "os_version": "$(uname -r)",
  "arch": "$(uname -m)",
  "model_file": "$(basename "$MODEL")",
  "model_size_bytes": $(stat -f%z "$MODEL" 2>/dev/null || stat --printf="%s" "$MODEL" 2>/dev/null || echo 0),
  "hardware": {
    "cpu_brand": "$([ "$PLATFORM" = "Darwin" ] && sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo 'unknown')",
    "ram_gb": $([ "$PLATFORM" = "Darwin" ] && sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f", $1/1073741824}' || grep MemTotal /proc/meminfo 2>/dev/null | awk '{printf "%.0f", $2/1048576}' || echo 0),
    "gpu_family": "$([ "$PLATFORM" = "Darwin" ] && echo "$GPU_INIT" | grep 'GPU family:' | head -1 | sed 's/.*: //' || echo 'N/A')",
    "has_tensor": $([ "$PLATFORM" = "Darwin" ] && (echo "$GPU_INIT" | grep -q 'has tensor.*true' && echo 'true' || echo 'false') || echo '"N/A"'),
    "apple_silicon": $([ "$PLATFORM" = "Darwin" ] && (sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -q Apple && echo 'true' || echo 'false') || echo 'false')
  }
}
HWEOF

zip -j "$ZIPFILE" "$PROFILE" 2>/dev/null
rm -f "$PROFILE"

echo ""
echo "============================================"
echo "  DIAGNOSTIC PACKAGE READY"
echo "============================================"
echo ""
echo "  Zip file: $(pwd)/$ZIPFILE"
echo "  Contents:"
zip -l "$ZIPFILE" 2>/dev/null | grep -E "\.txt|\.json" | sed 's/^/    /'
echo ""
echo "  Send this zip file to the TurboQuant team."
echo ""
