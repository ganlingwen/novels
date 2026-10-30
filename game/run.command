#!/bin/bash
set -euo pipefail

game_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
godot_bin="${GODOT_BIN:-}"
if [[ -z "$godot_bin" ]]; then
    if command -v godot >/dev/null 2>&1; then
        godot_bin="$(command -v godot)"
    elif [[ -x /Applications/Godot.app/Contents/MacOS/Godot ]]; then
        godot_bin="/Applications/Godot.app/Contents/MacOS/Godot"
    elif [[ -x "$HOME/Library/Caches/graymist-tools/Godot.app/Contents/MacOS/Godot" ]]; then
        godot_bin="$HOME/Library/Caches/graymist-tools/Godot.app/Contents/MacOS/Godot"
    else
        printf '%s\n' "需要 Godot 4.5 或更新的 Godot 4：安装后打开 game/project.godot，按 F6 运行 main.tscn。"
        printf '%s\n' "官方下载：https://godotengine.org/download/"
        exit 1
    fi
fi
exec "$godot_bin" --path "$game_dir" "$@"
