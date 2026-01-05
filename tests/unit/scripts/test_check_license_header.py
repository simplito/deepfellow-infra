# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for check_license_header.py module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_license_header import (
    check_file_header,
    extract_content_after_preamble,
    extract_header_year,
    find_python_files,
    fix_file_header,
    get_current_year,
    get_license_header,
    normalize_header,
    parse_gitignore,
    should_exclude,
)


@pytest.fixture
def current_year() -> int:
    """Current year for tests."""
    return get_current_year()


@pytest.fixture
def license_header(current_year: int) -> str:
    """License header with current year."""
    return get_license_header(current_year)


@pytest.fixture
def valid_content(license_header: str) -> str:
    """Valid Python file content with license header."""
    return f"{license_header}\n\nimport sys\n"


@pytest.fixture
def valid_with_shebang(license_header: str) -> str:
    """Valid Python file with shebang and license header."""
    return f"#!/usr/bin/env python3\n\n{license_header}\n\nimport sys\n"


@pytest.fixture
def valid_with_encoding(license_header: str) -> str:
    """Valid Python file with encoding declaration and license header."""
    return f"# -*- coding: utf-8 -*-\n\n{license_header}\n\nimport sys\n"


@pytest.fixture
def valid_with_both(license_header: str) -> str:
    """Valid Python file with shebang, encoding, and license header."""
    return f"#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n{license_header}\n\nimport sys\n"


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Sample project structure for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main")
    (tmp_path / "src" / "utils.py").write_text("# utils")
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "main.cpython-311.pyc").write_text("")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("# test")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib.py").write_text("# venv")
    return tmp_path


# --- get_license_header ---


def test_get_license_header_uses_current_year(current_year: int) -> None:
    header = get_license_header()
    assert f"Copyright © {current_year}" in header


def test_get_license_header_with_specific_year() -> None:
    header = get_license_header(2020)
    assert "Copyright © 2020" in header


# --- extract_header_year ---


def test_extract_header_year_finds_year(current_year: int, license_header: str) -> None:
    content = f"{license_header}\nimport sys\n"
    assert extract_header_year(content) == current_year


def test_extract_header_year_finds_old_year() -> None:
    header = get_license_header(2020)
    content = f"{header}\nimport sys\n"
    assert extract_header_year(content) == 2020


def test_extract_header_year_no_header() -> None:
    content = "import sys\n"
    assert extract_header_year(content) is None


def test_extract_header_year_with_shebang(current_year: int, license_header: str) -> None:
    content = f"#!/usr/bin/env python3\n\n{license_header}\nimport sys\n"
    assert extract_header_year(content) == current_year


# --- normalize_header ---


def test_normalize_header_removes_trailing_whitespace() -> None:
    assert normalize_header("line1   \nline2  ") == "line1\nline2"


def test_normalize_header_strips_outer_whitespace() -> None:
    assert normalize_header("\n\n  hello  \n\n") == "hello"


def test_normalize_header_preserves_inner_structure() -> None:
    assert normalize_header("line1\n\nline3") == "line1\n\nline3"


# --- extract_content_after_preamble ---


def test_extract_content_after_preamble_no_preamble() -> None:
    content = "# License\nimport sys"
    assert extract_content_after_preamble(content) == content


def test_extract_content_after_preamble_with_shebang() -> None:
    content = "#!/usr/bin/env python3\n\n# License"
    assert extract_content_after_preamble(content) == "# License"


def test_extract_content_after_preamble_with_encoding() -> None:
    content = "# -*- coding: utf-8 -*-\n\n# License"
    assert extract_content_after_preamble(content) == "# License"


def test_extract_content_after_preamble_with_shebang_and_encoding() -> None:
    content = "#!/usr/bin/env python3\n# coding: utf-8\n\n# License"
    assert extract_content_after_preamble(content) == "# License"


# --- parse_gitignore ---


def test_parse_gitignore_parses_patterns(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__\n.venv\n*.pyc\nbuild/\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"__pycache__", ".venv", "*.pyc", "build"})


def test_parse_gitignore_ignores_comments(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("# This is a comment\n__pycache__\n# Another comment\n.venv\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"__pycache__", ".venv"})


def test_parse_gitignore_ignores_empty_lines(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__\n\n\n.venv\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"__pycache__", ".venv"})


def test_parse_gitignore_strips_slashes(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("/build/\n/dist\nvenv/\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"build", "dist", "venv"})


def test_parse_gitignore_returns_empty_if_no_gitignore(tmp_path: Path) -> None:
    assert parse_gitignore(tmp_path) == frozenset()


def test_parse_gitignore_handles_complex_gitignore(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("""
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Environments
.venv/
venv/

# IDE
.idea/
.vscode/
""")

    patterns = parse_gitignore(tmp_path)

    assert "__pycache__" in patterns
    assert "*.py[cod]" in patterns
    assert "*.egg-info" in patterns
    assert ".venv" in patterns


# --- should_exclude ---


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        (Path("src/__pycache__/module.pyc"), "__pycache__", True),
        (Path("project/.venv/lib/site.py"), ".venv", True),
        (Path("pkg.egg-info/PKG-INFO"), "*.egg-info", True),
        (Path("src/mymodule/main.py"), "__pycache__", False),
        (Path("tests/test_main.py"), ".venv", False),
        (Path("src/cache.pyc"), "*.pyc", True),
        (Path("tests/test_utils.py"), "test_*", True),
    ],
)
def test_should_exclude_patterns(path: Path, pattern: str, expected: bool) -> None:
    assert should_exclude(path, frozenset({pattern})) is expected


def test_should_exclude_multiple_excludes() -> None:
    path = Path("src/__pycache__/cached.pyc")
    assert should_exclude(path, frozenset({"__pycache__", ".venv"})) is True


def test_should_exclude_no_excludes() -> None:
    assert should_exclude(Path("src/main.py"), frozenset()) is False


# --- check_file_header ---


def test_check_file_header_valid(tmp_path: Path, valid_content: str) -> None:
    file = tmp_path / "valid.py"
    file.write_text(valid_content)
    has_valid, year = check_file_header(file)
    assert has_valid is True
    assert year == get_current_year()


def test_check_file_header_valid_with_shebang(tmp_path: Path, valid_with_shebang: str) -> None:
    file = tmp_path / "with_shebang.py"
    file.write_text(valid_with_shebang)
    has_valid, _year = check_file_header(file)
    assert has_valid is True


def test_check_file_header_valid_with_encoding(tmp_path: Path, valid_with_encoding: str) -> None:
    file = tmp_path / "with_encoding.py"
    file.write_text(valid_with_encoding)
    has_valid, _year = check_file_header(file)
    assert has_valid is True


def test_check_file_header_valid_with_shebang_and_encoding(tmp_path: Path, valid_with_both: str) -> None:
    file = tmp_path / "with_both.py"
    file.write_text(valid_with_both)
    has_valid, _year = check_file_header(file)
    assert has_valid is True


def test_check_file_header_invalid_no_header(tmp_path: Path) -> None:
    file = tmp_path / "no_header.py"
    file.write_text("import sys\n")
    has_valid, year = check_file_header(file)
    assert has_valid is False
    assert year is None


def test_check_file_header_invalid_wrong_header(tmp_path: Path) -> None:
    file = tmp_path / "wrong_header.py"
    file.write_text("# MIT License\nimport sys\n")
    has_valid, year = check_file_header(file)
    assert has_valid is False
    assert year is None


def test_check_file_header_invalid_old_year(tmp_path: Path) -> None:
    """File with valid header but outdated year."""
    old_header = get_license_header(2020)
    file = tmp_path / "old_year.py"
    file.write_text(f"{old_header}\nimport sys\n")
    has_valid, year = check_file_header(file)
    assert has_valid is False
    assert year == 2020


def test_check_file_header_empty_file(tmp_path: Path) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")
    has_valid, year = check_file_header(file)
    assert has_valid is True
    assert year is None


def test_check_file_header_whitespace_only(tmp_path: Path) -> None:
    file = tmp_path / "whitespace.py"
    file.write_text("   \n\n")
    has_valid, _year = check_file_header(file)
    assert has_valid is True


def test_check_file_header_nonexistent_file(tmp_path: Path) -> None:
    has_valid, year = check_file_header(tmp_path / "nonexistent.py")
    assert has_valid is False
    assert year is None


# --- fix_file_header ---


def test_fix_file_header_without_header(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "no_header.py"
    file.write_text("import sys\n\ndef main():\n    pass\n")

    result = fix_file_header(file)

    assert result is True
    has_valid, _year = check_file_header(file)
    assert has_valid is True
    content = file.read_text()
    assert f"Copyright © {current_year}" in content
    assert "import sys" in content


def test_fix_file_header_with_shebang(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "with_shebang.py"
    file.write_text("#!/usr/bin/env python3\n\nimport sys\n")

    fix_file_header(file)

    has_valid, _ = check_file_header(file)
    assert has_valid is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n")
    assert f"Copyright © {current_year}" in content
    assert "import sys" in content


def test_fix_file_header_with_encoding(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "with_encoding.py"
    file.write_text("# -*- coding: utf-8 -*-\n\nimport sys\n")

    fix_file_header(file)

    has_valid, _ = check_file_header(file)
    assert has_valid is True
    content = file.read_text()
    assert content.startswith("# -*- coding: utf-8 -*-\n")
    assert f"Copyright © {current_year}" in content


def test_fix_file_header_with_shebang_and_encoding(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "with_both.py"
    file.write_text("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nimport sys\n")

    fix_file_header(file)

    has_valid, _ = check_file_header(file)
    assert has_valid is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n")
    assert f"Copyright © {current_year}" in content


def test_fix_file_header_empty_file(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "empty.py"
    file.write_text("")

    fix_file_header(file)

    content = file.read_text()
    assert f"Copyright © {current_year}" in content


def test_fix_file_header_updates_old_year(tmp_path: Path, current_year: int) -> None:
    """Fix should update year in existing header."""
    old_header = get_license_header(2020)
    file = tmp_path / "old_year.py"
    file.write_text(f"{old_header}\nimport sys\n")

    result = fix_file_header(file)

    assert result is True
    has_valid, year = check_file_header(file)
    assert has_valid is True
    assert year == current_year
    content = file.read_text()
    assert f"Copyright © {current_year}" in content
    assert "Copyright © 2020" not in content
    assert "import sys" in content


def test_fix_file_header_preserves_code(tmp_path: Path) -> None:
    file = tmp_path / "code.py"
    file.write_text("import sys\n\ndef main():\n    print('hello')\n")

    fix_file_header(file)

    content = file.read_text()
    assert "import sys" in content
    assert "def main():" in content
    assert "print('hello')" in content


def test_fix_file_header_nonexistent_file(tmp_path: Path) -> None:
    assert fix_file_header(tmp_path / "nonexistent.py") is False


def test_fix_file_header_idempotent(tmp_path: Path, valid_content: str, current_year: int) -> None:
    file = tmp_path / "already_valid.py"
    file.write_text(valid_content)

    has_valid, _ = check_file_header(file)
    assert has_valid is True

    fix_file_header(file)

    has_valid, year = check_file_header(file)
    assert has_valid is True
    assert year == current_year


# --- find_python_files ---


def test_find_python_files_finds_all(project_dir: Path) -> None:
    excludes = frozenset({"__pycache__", ".venv"})
    files = find_python_files([project_dir], excludes)
    assert {f.name for f in files} == {"main.py", "utils.py", "test_main.py"}


def test_find_python_files_excludes_pycache(project_dir: Path) -> None:
    excludes = frozenset({"__pycache__"})
    files = find_python_files([project_dir], excludes)
    assert not any("__pycache__" in str(f) for f in files)


def test_find_python_files_excludes_venv(project_dir: Path) -> None:
    excludes = frozenset({".venv"})
    files = find_python_files([project_dir], excludes)
    assert not any(".venv" in str(f) for f in files)


def test_find_python_files_non_recursive(project_dir: Path) -> None:
    (project_dir / "root.py").write_text("# root")
    files = find_python_files([project_dir], frozenset(), recursive=False)
    assert len(files) == 1
    assert files[0].name == "root.py"


def test_find_python_files_specific_file(project_dir: Path) -> None:
    target = project_dir / "src" / "main.py"
    files = find_python_files([target], frozenset())
    assert files == [target]


def test_find_python_files_empty_directory(tmp_path: Path) -> None:
    assert find_python_files([tmp_path], frozenset()) == []


def test_find_python_files_respects_gitignore(project_dir: Path) -> None:
    gitignore = project_dir / ".gitignore"
    gitignore.write_text("__pycache__\n.venv\n")

    excludes = parse_gitignore(project_dir)
    files = find_python_files([project_dir], excludes)

    assert {f.name for f in files} == {"main.py", "utils.py", "test_main.py"}


# --- Year validation edge cases ---


def test_check_file_header_with_mocked_year(tmp_path: Path) -> None:
    """Test that year check uses current year dynamically."""
    header_2024 = get_license_header(2024)
    file = tmp_path / "test.py"
    file.write_text(f"{header_2024}\nimport sys\n")

    # Mock current year to be 2024
    with patch("scripts.check_license_header.get_current_year", return_value=2024):
        has_valid, year = check_file_header(file)
        assert has_valid is True
        assert year == 2024

    # Without mock, should fail (unless we're actually in 2024)
    actual_year = get_current_year()
    if actual_year != 2024:
        has_valid, year = check_file_header(file)
        assert has_valid is False
        assert year == 2024


def test_fix_file_header_updates_to_current_year_mocked(tmp_path: Path) -> None:
    """Test that fix updates to whatever the current year is."""
    header_2020 = get_license_header(2020)
    file = tmp_path / "old.py"
    file.write_text(f"{header_2020}\nimport sys\n")

    with patch("scripts.check_license_header.get_current_year", return_value=2030):
        fix_file_header(file)
        content = file.read_text()
        assert "Copyright © 2030" in content
        assert "Copyright © 2020" not in content
