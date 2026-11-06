"""Where everything lives, and how the notebooks find the code in src/.

1. **Folders.** Defined once here so no notebook writes paths by hand.

2. **Importing from src/.** There is deliberately no `__init__.py`: the code is
   meant to be read one file at a time, not installed as a package. Python
   only finds modules that sit directly on `sys.path`, so
   `add_src_subfolders_to_path()` adds `corpus/`, `index/` and `analysis/` to it.

Usage, at the top of every notebook:

    sys.path.insert(0, str(Path.cwd().parent / "src"))
    import paths
    paths.add_src_subfolders_to_path()
    from chunking import create_document_chunks

Because every module is imported by its bare name, two modules in different
subfolders must never share a filename.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

# Downloaded PDFs. Not included: notebooks 01 and 02 download them again.
RAW_DIR = PROJECT_DIR / "data" / "raw" / "banxico"

# Everything the pipeline produces: pages, documents, chunks, indices, LLM scores.
PROCESSED_DIR = PROJECT_DIR / "data" / "processed" / "banxico"

# Tables (CSV) and figures that notebook 06 produces.
RESULTS_DIR = PROJECT_DIR / "resultados"
FIGURES_DIR = RESULTS_DIR / "figuras"

SRC_SUBFOLDERS = ["corpus", "index", "analysis"]


def add_src_subfolders_to_path():
    """Make `from <module> import ...` work for modules inside the src/ subfolders."""
    for subfolder in SRC_SUBFOLDERS:
        folder = str(SRC_DIR / subfolder)
        if folder not in sys.path:
            sys.path.insert(0, folder)
