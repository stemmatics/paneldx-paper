"""Build a self-contained LaTeX bundle for Overleaf and for journal submission."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "submission"
FIGURES = [
    "figure1_null_separation.pdf",
    "figure2_domain_dependence.pdf",
    "figure3_concealment.pdf",
]


def main() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "figures").mkdir(parents=True)

    source = (ROOT / "tex" / "manuscript.tex").read_text()
    source = source.replace("../figures/", "figures/")
    (BUILD / "manuscript.tex").write_text(source)

    for name in FIGURES:
        shutil.copy2(ROOT / "figures" / name, BUILD / "figures" / name)

    archive = ROOT / "build" / "manuscript-submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(BUILD.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD))

    print(f"wrote {archive.relative_to(ROOT)}  ({archive.stat().st_size // 1024} KB)")
    for name in zipfile.ZipFile(archive).namelist():
        print(f"  {name}")


if __name__ == "__main__":
    main()
