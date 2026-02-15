"""Export a draw.io diagram to PNG if a draw.io CLI is available."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_drawio_exe() -> str | None:
    candidates = ["drawio", "draw.io", "drawio.exe", "draw.io.exe"]
    for name in candidates:
        exe = shutil.which(name)
        if exe:
            return exe

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    known_paths = [
        os.path.join(program_files, "draw.io", "draw.io.exe"),
        os.path.join(program_files_x86, "draw.io", "draw.io.exe"),
        os.path.join(program_files, "diagrams.net", "diagrams.net.exe"),
        os.path.join(program_files_x86, "diagrams.net", "diagrams.net.exe"),
    ]
    for path in known_paths:
        if os.path.exists(path):
            return path
    return None


def export_drawio_png(input_path: Path, output_path: Path) -> bool:
    exe = find_drawio_exe()
    if exe is None:
        print("draw.io CLI not found.")
        print("Manual export: open the .drawio file in draw.io -> File -> Export as -> PNG.")
        return False

    cmd = [
        exe,
        "--export",
        "--format",
        "png",
        "--output",
        output_path.as_posix(),
        input_path.as_posix(),
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"draw.io export failed: {exc}")
        print("Manual export: open the .drawio file in draw.io -> File -> Export as -> PNG.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Export draw.io to PNG")
    parser.add_argument("--input", type=str, default=os.path.join("assets", "moe_routing.drawio"))
    parser.add_argument("--output", type=str, default=os.path.join("assets", "moe_routing.png"))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = export_drawio_png(input_path, output_path)
    if ok:
        print(f"Exported: {output_path}")


if __name__ == "__main__":
    main()
