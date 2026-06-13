"""
Standalone test script for textual_image renderable.

Usage:
    uv run python test_image.py path/to/your/image.png

    Or with an image in .agentcrew/images/:
    uv run python test_image.py .agentcrew/images/generated_*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python test_image.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"❌ File not found: {image_path}")
        sys.exit(1)

    try:
        from textual_image.renderable import Image as TextualImage
    except ImportError:
        print("❌ textual-image is not installed.")
        print("   Install: uv add textual-image")
        sys.exit(1)

    console = Console()
    console.print(f"\n📷 Displaying: [bold cyan]{image_path}[/bold cyan]\n")
    console.print(TextualImage(str(image_path)))
    console.print("\n✅ Done\n")


if __name__ == "__main__":
    main()
