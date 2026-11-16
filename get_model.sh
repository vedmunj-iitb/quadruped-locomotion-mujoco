#!/usr/bin/env bash
# Downloads the Unitree Go2 model from MuJoCo Menagerie (BSD-3-Clause licence) into third_party/.
# Only the unitree_go2 folder is fetched, at a fixed commit, so every run gets the same model.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=370c43965323c0944b433b202aac5058b5f907a9
DIR=third_party/mujoco_menagerie
if [ ! -f "$DIR/unitree_go2/go2.xml" ]; then
  rm -rf "$DIR"
  git clone -q --filter=blob:none --sparse --no-checkout \
    https://github.com/google-deepmind/mujoco_menagerie.git "$DIR"
  git -C "$DIR" sparse-checkout set unitree_go2
  git -C "$DIR" checkout -q "$COMMIT"
fi
echo "model ready: $DIR/unitree_go2"
