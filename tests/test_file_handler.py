"""
Test script for FileHandler.process_file() with various input file types.

Manual mode — process your own file and see the full converted output:
    uv run python tests/test_file_handler.py path/to/your/file.pdf
    uv run python tests/test_file_handler.py ~/Documents/report.docx
    uv run python tests/test_file_handler.py ./image.png

Auto mode — generate sample files and show a summary table:
    uv run python tests/test_file_handler.py

Run with pytest:
    uv run pytest tests/test_file_handler.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from AgentCrew.modules.utils.file_handler import FileHandler

console = Console()


def _make_image(path: Path) -> None:
    """Create a small valid PNG using Pillow."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(255, 100, 50))
    img.save(str(path), format="PNG")


def _make_webp(path: Path) -> None:
    """Create a small valid WebP using Pillow."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(50, 200, 100))
    img.save(str(path), format="WEBP", quality=80)


def build_test_files(tmp_dir: Path) -> list[tuple[str, Path, str]]:
    """
    Create various test files inside *tmp_dir*.

    Returns a list of (label, file_path, description) tuples.
    """
    files: list[tuple[str, Path, str]] = []

    # 1. Plain text file
    txt_path = tmp_dir / "sample.txt"
    txt_path.write_text(
        "Hello, this is a plain text file.\nLine two here.", encoding="utf-8"
    )
    files.append(("text/plain (.txt)", txt_path, "Should return text content"))

    # 2. Python source file
    py_path = tmp_dir / "snippet.py"
    py_path.write_text('print("hello world")\n', encoding="utf-8")
    files.append(("text/x-python (.py)", py_path, "Should return text content"))

    # 3. JSON file (application/json is not in ALLOWED_MIME_TYPES and not text/*, so it is rejected)
    json_path = tmp_dir / "data.json"
    json_path.write_text(
        json.dumps({"key": "value", "num": 42}, indent=2), encoding="utf-8"
    )
    files.append(
        (
            "application/json (.json)",
            json_path,
            "Should return None (MIME not in allowed list)",
        )
    )

    # 4. Markdown file
    md_path = tmp_dir / "readme.md"
    md_path.write_text("# Title\n\nSome **markdown** text.\n", encoding="utf-8")
    files.append(("text/markdown (.md)", md_path, "Should return text content"))

    # 5. Empty text file
    empty_path = tmp_dir / "empty.txt"
    empty_path.write_text("", encoding="utf-8")
    files.append(
        ("empty text file", empty_path, "Should return text with empty content")
    )

    # 6. PNG image
    png_path = tmp_dir / "test.png"
    _make_image(png_path)
    files.append(
        ("image/png (.png)", png_path, "Should return image_url with base64 data")
    )

    # 7. WebP image
    webp_path = tmp_dir / "test.webp"
    _make_webp(webp_path)
    files.append(
        ("image/webp (.webp)", webp_path, "Should return image_url with base64 data")
    )

    # 8. Non-existent file
    missing_path = tmp_dir / "does_not_exist.txt"
    files.append(
        ("non-existent file", missing_path, "Should return None (validation fails)")
    )

    # 9. Unsupported binary file (.bin)
    bin_path = tmp_dir / "data.bin"
    bin_path.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    files.append(
        ("unsupported binary (.bin)", bin_path, "Should return None (unsupported MIME)")
    )

    # 10. Oversized file (> 50 MB) — create a sparse file without actually writing 50 MB
    big_path = tmp_dir / "huge.txt"
    big_path.write_text("x", encoding="utf-8")
    # Force the file size to just over the 50 MB limit using truncate
    with open(big_path, "r+b") as f:
        f.truncate(50 * 1024 * 1024 + 1)
    files.append(
        ("oversized file (>50MB)", big_path, "Should return None (size limit exceeded)")
    )

    # 11. PDF file (only if docling is available, otherwise still test the path)
    pdf_path = tmp_dir / "sample.pdf"
    try:
        # Create a minimal valid PDF using reportlab if available
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "This is a test PDF document.")
            c.save()
            files.append(
                (
                    "application/pdf (.pdf)",
                    pdf_path,
                    "Should return markdown via Docling",
                )
            )
        except ImportError:
            # Write a dummy .pdf — Docling will fail, handler returns None
            pdf_path.write_bytes(b"%PDF-1.4 dummy")
            files.append(
                (
                    "application/pdf (.pdf, dummy)",
                    pdf_path,
                    "Docling may fail; should return None or markdown",
                )
            )
    except ImportError:
        # Docling not installed — write a dummy PDF to test the fallback path
        pdf_path.write_bytes(b"%PDF-1.4 dummy")
        files.append(
            (
                "application/pdf (.pdf, no docling)",
                pdf_path,
                "No Docling; should return None",
            )
        )

    return files


def _summarise_result(result: dict | None) -> str:
    """Produce a human-readable summary of the process_file return value."""
    if result is None:
        return "None (rejected or unsupported)"

    rtype = result.get("type", "?")
    if rtype == "text":
        text = result.get("text", "")
        preview = text[:200].replace("\n", "\\n")
        suffix = "..." if len(text) > 200 else ""
        return f'type="text", len={len(text)}, preview="{preview}{suffix}"'
    elif rtype == "image_url":
        url = result.get("image_url", {}).get("url", "")
        detail = result.get("image_url", {}).get("detail", "")
        # Truncate the base64 data URI for display
        url_preview = url[:80] + "..." if len(url) > 80 else url
        return f'type="image_url", detail="{detail}", url_len={len(url)}, url_preview="{url_preview}"'
    else:
        return f'type="{rtype}", keys={list(result.keys())}'


def run_interactive() -> None:
    """Create temp files, run FileHandler on each, and print results in a table."""
    handler = FileHandler()

    with tempfile.TemporaryDirectory(prefix="filehandler_test_") as tmp:
        tmp_dir = Path(tmp)
        test_files = build_test_files(tmp_dir)

        console.print(
            Panel.fit(
                f"[bold cyan]FileHandler.process_file()[/bold cyan] — {len(test_files)} test cases\n"
                f"Temp dir: {tmp_dir}",
                border_style="cyan",
            )
        )

        table = Table(show_header=True, header_style="bold magenta", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Label", style="cyan", no_wrap=True)
        table.add_column("File", style="dim", overflow="fold")
        table.add_column("Expected", style="yellow", overflow="fold")
        table.add_column("Result Summary", style="green", overflow="fold")

        for idx, (label, fpath, expected) in enumerate(test_files, 1):
            result = handler.process_file(str(fpath))
            summary = _summarise_result(result)
            table.add_row(str(idx), label, str(fpath), expected, summary)

        console.print(table)

        # Detailed output for text-type results
        console.print("\n[bold]--- Detailed text outputs ---[/bold]\n")
        for idx, (label, fpath, _) in enumerate(test_files, 1):
            result = handler.process_file(str(fpath))
            if result and result.get("type") == "text":
                text = result.get("text", "")
                console.print(
                    Panel(
                        text[:500] + ("..." if len(text) > 500 else ""),
                        title=f"#{idx} {label}",
                        border_style="blue",
                    )
                )

        console.print("\n[bold green]✅ Test complete.[/bold green]")


# --------------------------------------------------------------------------- #
# Pytest-compatible tests                                                     #
# --------------------------------------------------------------------------- #


def test_text_file_returns_text_content(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "hello.txt"
    fpath.write_text("Hello world", encoding="utf-8")

    result = handler.process_file(str(fpath))

    assert result is not None
    assert result["type"] == "text"
    assert "Hello world" in result["text"]


def test_python_file_returns_text_content(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "code.py"
    fpath.write_text('print("hi")', encoding="utf-8")

    result = handler.process_file(str(fpath))

    assert result is not None
    assert result["type"] == "text"
    assert 'print("hi")' in result["text"]


def test_json_file_returns_none(tmp_path: Path) -> None:
    """application/json is not in ALLOWED_MIME_TYPES and doesn't start with text/.

    The handler's validate_file() rejects it, so process_file returns None.
    """
    handler = FileHandler()
    fpath = tmp_path / "data.json"
    fpath.write_text('{"a": 1}', encoding="utf-8")

    result = handler.process_file(str(fpath))

    assert result is None


def test_nonexistent_file_returns_none(tmp_path: Path) -> None:
    handler = FileHandler()
    result = handler.process_file(str(tmp_path / "nope.txt"))
    assert result is None


def test_unsupported_binary_returns_none(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "data.bin"
    fpath.write_bytes(b"\x00\x01\x02")
    result = handler.process_file(str(fpath))
    assert result is None


def test_oversized_file_returns_none(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "big.txt"
    fpath.write_text("x", encoding="utf-8")
    with open(fpath, "r+b") as f:
        f.truncate(50 * 1024 * 1024 + 1)
    result = handler.process_file(str(fpath))
    assert result is None


def test_png_image_returns_image_url(tmp_path: Path) -> None:
    from PIL import Image

    handler = FileHandler()
    fpath = tmp_path / "img.png"
    Image.new("RGB", (32, 32), (200, 50, 50)).save(str(fpath), format="PNG")

    result = handler.process_file(str(fpath))

    assert result is not None
    assert result["type"] == "image_url"
    assert result["image_url"]["url"].startswith("data:")
    assert result["image_url"]["detail"] == "high"


def test_empty_text_file_returns_text(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "empty.txt"
    fpath.write_text("", encoding="utf-8")

    result = handler.process_file(str(fpath))

    assert result is not None
    assert result["type"] == "text"


def test_validate_file_nonexistent() -> None:
    handler = FileHandler()
    assert handler.validate_file("/nonexistent/path/file.txt") is False


def test_validate_file_supported_text(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "ok.txt"
    fpath.write_text("ok", encoding="utf-8")
    assert handler.validate_file(str(fpath)) is True


def test_validate_file_unsupported_mime(tmp_path: Path) -> None:
    handler = FileHandler()
    fpath = tmp_path / "weird.xyz"
    fpath.write_bytes(b"data")
    assert handler.validate_file(str(fpath)) is False


def test_guess_mime_by_extension() -> None:
    assert FileHandler.guess_mime_by_extension("doc.pdf") == "application/pdf"
    assert FileHandler.guess_mime_by_extension("image.webp") == "image/webp"
    assert FileHandler.guess_mime_by_extension("file.xyz") is None


def run_manual(file_path: str) -> None:
    """Process a single user-provided file and print the full converted output."""
    handler = FileHandler()
    resolved = os.path.expanduser(file_path)

    console.print(
        Panel.fit(
            f"[bold cyan]FileHandler.process_file()[/bold cyan] — manual mode\n"
            f"Input: {resolved}",
            border_style="cyan",
        )
    )

    # Show file metadata
    if os.path.exists(resolved):
        size = os.path.getsize(resolved)
        size_str = (
            f"{size:,} bytes ({size / 1024:.1f} KB)"
            if size < 1024 * 1024
            else f"{size:,} bytes ({size / (1024 * 1024):.1f} MB)"
        )
        mime_type, _ = __import__("mimetypes").guess_type(resolved)
        if not mime_type:
            mime_type = handler.guess_mime_by_extension(resolved)
        console.print(f"  [dim]File size:[/dim] {size_str}")
        console.print(f"  [dim]MIME type:[/dim] {mime_type or 'unknown'}")
        console.print(
            f"  [dim]Valid:    [/dim] {'✅ yes' if handler.validate_file(resolved) else '❌ no'}"
        )
    else:
        console.print(f"  [red]❌ File does not exist: {resolved}[/red]")

    console.print()

    # Process and display
    result = handler.process_file(resolved)

    if result is None:
        console.print(
            Panel(
                "[red]process_file() returned None[/red]\n\n"
                "This means the file was rejected. Common reasons:\n"
                "  • File doesn't exist\n"
                "  • File exceeds 50 MB limit\n"
                "  • MIME type not in ALLOWED_MIME_TYPES and not text/*\n"
                "  • Docling failed to convert the document",
                title="Result",
                border_style="red",
            )
        )
        return

    rtype = result.get("type", "?")

    if rtype == "text":
        text = result.get("text", "")
        console.print('  [green]type[/green]     = "text"')
        console.print(f"  [green]text length[/green] = {len(text):,} chars")
        console.print()
        console.print(
            Panel(
                text,
                title="Full converted content",
                border_style="green",
            )
        )

    elif rtype == "image_url":
        url = result.get("image_url", {}).get("url", "")
        detail = result.get("image_url", {}).get("detail", "")
        console.print('  [green]type[/green]   = "image_url"')
        console.print(f"  [green]detail[/green] = {detail}")
        console.print(f"  [green]url[/green]    = data URI ({len(url):,} chars)")
        console.print()
        # Show a truncated preview of the base64 data
        preview = url[:120] + "..." if len(url) > 120 else url
        console.print(
            Panel(
                f"[dim]{preview}[/dim]",
                title="Image data URI (truncated)",
                border_style="green",
            )
        )
        console.print(
            "\n[yellow]💡 The full base64 data URI is ready to send to an LLM.\n"
            "   It's truncated here for display, but the complete data is in the result dict.[/yellow]"
        )

    else:
        console.print(
            Panel(
                f'type = "{rtype}"\n\nkeys = {list(result.keys())}\n\n{result}',
                title="Result (unknown type)",
                border_style="yellow",
            )
        )

    console.print("\n[bold green]✅ Done.[/bold green]")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        run_manual(sys.argv[1])
    else:
        run_interactive()
