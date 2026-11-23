#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
echo "$SCRIPT_DIR -- script dir"
AMIE_DIR="$(cd "$SCRIPT_DIR/../amieCli" && pwd -P)"

LOG_FILE="${LOG_FILE:-/tmp/aoti-setup.log}"

export AOTI_FFI_LIB="$SCRIPT_DIR/onnx_aoti/kotlin/src/main/resources/native/macos-aarch64/libaoti_ffi.dylib"
export AOTI_PYTHON="$SCRIPT_DIR/env/bin/python"
export AOTI_COMPILER_SCRIPT="$SCRIPT_DIR/onnx_aoti/onnx_aoti.py"

if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
    C_RESET=$'\033[0m'
    C_BLUE=$'\033[1;34m'
    C_GREEN=$'\033[0;32m'
    C_YELLOW=$'\033[0;33m'
    C_RED=$'\033[0;31m'
else
    C_RESET='' C_BLUE='' C_GREEN='' C_YELLOW='' C_RED=''
fi

step()  { printf '%s==> %s%s\n'      "$C_BLUE"   "$*" "$C_RESET" >&2; }
info()  { printf '%s[INFO]%s  %s\n'   "$C_BLUE"   "$C_RESET" "$*" >&2; }
ok()    { printf '%s[OK]%s    %s\n'   "$C_GREEN"  "$C_RESET" "$*" >&2; }
warn()  { printf '%s[WARN]%s  %s\n'   "$C_YELLOW" "$C_RESET" "$*" >&2; }
die()   { printf '%s[ERROR]%s %s\n'   "$C_RED"    "$C_RESET" "$*" >&2; exit 1; }

writeLog() {
    local level="$1"
    shift
    local ts project
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    project="${PROJECT:-${PWD##*/}}"

    printf '%-7s [%s] %s\n' "[$level]" "$project" "$*" >&2
    printf '%s %-7s [%s] %s\n' "$ts" "[$level]" "$project" "$*" >> "$LOG_FILE"
}

consoleLog() {
    if [[ "${1:-}" == "-S" ]]; then
        local val="${2:-}"
        local label="${3:-Target}"
        if [[ -n "$val" ]]; then
            writeLog "OK" "$label: $val"
            return 0
        else
            writeLog "ERROR" "$label not found (dir: $PWD)"
            return 1
        fi
    fi

    local cmd="$*"
    local output rc=0
    output=$("$@" 2>&1) || rc=$?

    if [[ -n "$output" ]]; then
        printf '%s\n' "$output" >&2
        printf '%s\n' "$output" >> "$LOG_FILE"
    fi

    if (( rc == 0 )); then
        writeLog "OK" "$cmd"
    else
        writeLog "ERROR" "$cmd (exit code: $rc, dir: $PWD)"
        return "$rc"
    fi
}

provisioned() {
	cd "$AMIE_DIR" || return 1
	consoleLog bin/amie examples/aoti-vector/compare.kts
}

provisionedLibraryApi() {
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

    cd "$AMIE_DIR" 
    consoleLog bin/amie /tmp/test_aoti_default.kts
}

provisioned
provisionedLibraryApi
