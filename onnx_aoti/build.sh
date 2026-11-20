#!/usr/bin/env bash
# Builds the Rust library, bundles it into the Kotlin resources, and packages the Kotlin jar.
# Requires LIBTORCH (the libtorch root, containing lib/ and include/) and kotlinc.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
: "${LIBTORCH:?set LIBTORCH to the libtorch root (contains lib/ and include/)}"

case "$(uname -s)" in
    Darwin) os=macos; lib=libaoti_ffi.dylib ;;
    Linux)  os=linux; lib=libaoti_ffi.so ;;
    *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
    arm64|aarch64) arch=aarch64 ;;
    x86_64|amd64)  arch=x86_64 ;;
    *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac
platform="$os-$arch"

(cd "$here/rust" && cargo build --release)

resources="$here/kotlin/src/main/resources/native/$platform"
mkdir -p "$resources"
cp "$here/rust/target/release/$lib" "$resources/$lib"

cd "$here/kotlin"
kotlinc src/main/kotlin/*.kt -include-runtime -d aoti-demo.jar
jar uf aoti-demo.jar -C src/main/resources native
echo "built $here/kotlin/aoti-demo.jar with native/$platform/$lib"
