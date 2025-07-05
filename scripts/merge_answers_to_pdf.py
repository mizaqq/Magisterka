#!/usr/bin/env python3
"""Merge individual answer markdown files into a single PDF.

Usage:
    python scripts/merge_answers_to_pdf.py [-o OUTPUT_PDF]

The script performs the following steps:
1. Collects all markdown files in the `answers/` directory that follow the
   `NN_*.md` naming convention.
2. Sorts them numerically by the leading question number so the final document
   respects original ordering.
3. Concatenates their contents into a temporary markdown file with page breaks
   between questions.
4. Converts the merged markdown to PDF using `pandoc` (preferred) or `pypandoc`.

Requirements:
    * pandoc must be installed and in PATH. Install from https://pandoc.org.
    * For better typography, PandaTeX or another LaTeX engine should be
      available, but pandoc will fall back to its default PDF method if not.

Example:
    python scripts/merge_answers_to_pdf.py -o outputs/thesis_answers.pdf
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ANSWERS_DIR = Path("answers")
DEFAULT_OUTPUT = Path("outputs/all_answers.pdf")
PAGE_BREAK = "\n\\newpage\n\n"


def gather_markdown_files() -> list[Path]:
    """Return list of answer markdown files sorted by leading number."""
    pattern = re.compile(r"^(\d{2})_.*\.md$")
    files = []
    for md_file in ANSWERS_DIR.glob("*.md"):
        match = pattern.match(md_file.name)
        if match:
            files.append((int(match.group(1)), md_file))
    return [f for _, f in sorted(files)]


def merge_markdown(files: list[Path]) -> str:
    """Concatenate markdown files with page breaks."""
    merged_parts = []
    for path in files:
        merged_parts.append(path.read_text(encoding="utf-8"))
        merged_parts.append(PAGE_BREAK)
    return "".join(merged_parts)


def convert_markdown_to_pdf(markdown_text: str, output_pdf: Path) -> None:
    """Convert markdown text to PDF using pandoc (preferred)."""
    try:
        # Write markdown to temp file
        with NamedTemporaryFile("w", delete=False, suffix=".md", encoding="utf-8") as tmp_md:
            tmp_md.write(markdown_text)
            tmp_md_path = tmp_md.name

        # Use pandoc via subprocess for maximum compatibility
        result = subprocess.run(
            [
                "pandoc",
                "--from",
                "markdown",
                "--pdf-engine",
                "xelatex",
                "-o",
                str(output_pdf),
                tmp_md_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("[ERROR] pandoc failed:\n", result.stderr, file=sys.stderr)
            sys.exit(result.returncode)
        print(f"✅ PDF generated: {output_pdf}")
    finally:
        # Ensure temp file is removed
        try:
            Path(tmp_md_path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge answer markdown files into a PDF.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path of the PDF to create (default: outputs/all_answers.pdf)",
    )
    args = parser.parse_args()

    output_pdf: Path = args.output
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    md_files = gather_markdown_files()
    if not md_files:
        print("No markdown files found in 'answers/' directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(md_files)} markdown files...")
    merged_markdown = merge_markdown(md_files)

    print("Converting merged markdown to PDF via pandoc...")
    convert_markdown_to_pdf(merged_markdown, output_pdf)


if __name__ == "__main__":
    main()
