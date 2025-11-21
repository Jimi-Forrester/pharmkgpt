from pathlib import Path


def test_readme_exists():
    assert Path("README.md").is_file(), "README.md should exist in project root"

