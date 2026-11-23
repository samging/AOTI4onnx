#!/usr/bin/env bash
set -euo pipefail

if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'
    C_BLUE=$'\033[1;34m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'
else
    C_RESET='' C_BLUE='' C_GREEN='' C_YELLOW='' C_RED=''
fi

writeLog() {
    local level="$1"
    shift
    local ts project color=""
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    project="${PROJECT:-${PWD##*/}}"

    case "$level" in
        OK)    color="$C_GREEN" ;;
        INFO)  color="$C_BLUE" ;;
        WARN)  color="$C_YELLOW" ;;
        ERROR) color="$C_RED" ;;
    esac

    # Colored output in terminal (stderr)
    if [[ "$level" == "ERROR" ]]; then
        printf '%s%-7s%s [%s] %s (dir: %s)\n' "$color" "[$level]" "$C_RESET" "$project" "$*" "$PWD" >&2
        printf '%s %-7s [%s] %s (dir: %s)\n' "$ts" "[$level]" "$project" "$*" "$PWD" >> "$LOG_FILE"
    else
        printf '%s%-7s%s [%s] %s\n' "$color" "[$level]" "$C_RESET" "$project" "$*" >&2
        printf '%s %-7s [%s] %s\n' "$ts" "[$level]" "$project" "$*" >> "$LOG_FILE"
    fi
}

consoleLog() {
    if [[ "${1:-}" == "-S" ]]; then
        local val="${2:-}"
        local label="${3:-Target}"
        if [[ -n "$val" ]]; then
            writeLog "OK" "$label: $val"
            return 0
        else
            writeLog "ERROR" "$label not found"
            return 1
        fi
    fi

    local cmd="$*"
    local output rc=0
    output=$("$@" 2>&1) || rc=$?

    if (( rc == 0 )); then
        [[ -n "$output" ]] && printf '%s\n' "$output" >> "$LOG_FILE"
        writeLog "OK" "$cmd"
    else
        if [[ -n "$output" ]]; then
            printf '%s\n' "$output" >&2
            printf '%s\n' "$output" >> "$LOG_FILE"
        fi
        writeLog "ERROR" "$cmd (exit code: $rc)"
        return "$rc"
    fi
}

provisioned() (
    consoleLog -S "$AMIE_DIR" "AMIE_DIR"
    cd "$AMIE_DIR" || return 1
    consoleLog bin/amie examples/aoti-vector/compare.kts
)

provisionedLibraryApi() (
    consoleLog -S "$AMIE_DIR" "AMIE_DIR"
    cd "$AMIE_DIR" || return 1

    local onnx_file
    onnx_file="$(find "examples/aoti-vector" -maxdepth 1 -name '*.onnx' | head -n 1)"
    consoleLog -S "$onnx_file" "Model"

    cat << EOF > /tmp/test_aoti_default.kts
val modelPath = "$onnx_file"

val defaultMod = initMod(modelPath)
println("Default runtime answer: " + defaultMod.answer("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8"))

val ortMod = initMod(modelPath, "ort")
println("Explicit ORT answer:    " + ortMod.answer("0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8"))
EOF

    consoleLog bin/amie /tmp/test_aoti_default.kts
)

resolve_amie_dir() {
    local candidate="${1:-}"
    if [[ -n "$candidate" && -x "$candidate/bin/amie" ]]; then
        (cd "$candidate" && pwd -P)
    elif [[ -n "${AMIE_DIR:-}" && -x "$AMIE_DIR/bin/amie" ]]; then
        (cd "$AMIE_DIR" && pwd -P)
    elif [[ -x "$PWD/bin/amie" ]]; then
        pwd -P
    elif command -v amie >/dev/null 2>&1; then
        (cd "$(dirname "$(command -v amie)")/.." && pwd -P)
    elif [[ -x "$SCRIPT_DIR/../amieCli/bin/amie" ]]; then
        (cd "$SCRIPT_DIR/../amieCli" && pwd -P)
    fi
}

run() {
    local amieCliPath="${1:-}"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
    LOG_FILE="${LOG_FILE:-/tmp/aoti-setup.log}"
    AMIE_DIR="$(resolve_amie_dir "$amieCliPath" || true)"

    writeLog "INFO" "SCRIPT_DIR: $SCRIPT_DIR"

    export AOTI_FFI_LIB="$SCRIPT_DIR/onnx_aoti/kotlin/src/main/resources/native/macos-aarch64/libaoti_ffi.dylib"
    export AOTI_PYTHON="$SCRIPT_DIR/env/bin/python"
    export AOTI_COMPILER_SCRIPT="$SCRIPT_DIR/onnx_aoti/onnx_aoti.py"

    provisioned
    provisionedLibraryApi
}

run "$@"
