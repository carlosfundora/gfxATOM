#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/local/ai/projects/gfxATOM-Rust"
TARGET="$ROOT/vendor-crates"

mkdir -p "$TARGET"
rm -rf "$TARGET"/*

# Canonical crate collections to mirror into the fusion workspace.
rsync -a --delete /home/local/ai/projects/rust/crates/ "$TARGET/projects-rust-crates/"
rsync -a --delete /home/local/ai/projects/harness/rust/crates/ "$TARGET/harness-rust-crates/"
rsync -a --delete /home/local/ai/projects/DATALORE/rust/crates/ "$TARGET/datalore-rust-crates/"
rsync -a --delete /home/local/ai/projects/DEMERZEL/rust/crates/ "$TARGET/demerzel-rust-crates/"
if [ -d /home/local/ai/projects/projects/rust/crates/harness ]; then
  rsync -a --delete /home/local/ai/projects/projects/rust/crates/harness/ "$TARGET/projects-mirror-harness-rust-crates/"
fi

echo "Rust crate mirror refreshed at $TARGET"
