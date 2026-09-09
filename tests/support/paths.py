"""Repository paths shared by tests without depending on the current directory."""

from pathlib import Path


TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
DOCS_ROOT = REPO_ROOT / "docs"
EXAMPLES_ROOT = REPO_ROOT / "examples"
RBLN_SMI_FIXTURES = TESTS_ROOT / "fixtures" / "rbln_smi"
