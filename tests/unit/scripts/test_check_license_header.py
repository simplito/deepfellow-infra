# DeepFellow Software Framework.
# Copyright © 2026 Simplito sp. z o.o.
#
# This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
# This software is Licensed under the DeepFellow Free License.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for check_license_header.py module."""

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_license_header import (
    apply_fixes,
    build_excludes,
    check_file_header,
    classify_files,
    create_parser,
    extract_content_after_preamble,
    extract_header_year,
    find_python_files,
    fix_file_header,
    get_current_year,
    get_files_to_check,
    get_license_header,
    main,
    normalize_header,
    parse_gitignore,
    report_issues,
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


@pytest.fixture
def gitignore_path(tmp_path: Path):
    """Fixture to write content to a .gitignore file in a temp directory."""
    return tmp_path / ".gitignore"


@pytest.mark.parametrize(("current_year", "ending"), [(None, ""), (2020, "2020")])
def test_get_license_header_uses_current_year(current_year: int | None, ending: str) -> None:
    header = get_license_header(current_year)

    assert " ".join(("Copyright ©", ending)) in header


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


@pytest.mark.parametrize(
    ("input_str", "expected_output"),
    [
        ("line1   \nline2  ", "line1\nline2"),
        ("\n\n  hello  \n\n", "hello"),
        ("line1\n\nline3", "line1\n\nline3"),
    ],
    ids=["trailing_whitespace", "outer_whitespace", "inner_structure"],
)
def test_normalize_header(input_str: str, expected_output: str) -> None:
    assert normalize_header(input_str) == expected_output


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("# License\nimport sys", "# License\nimport sys"),
        ("#!/usr/bin/env python3\n\n# License", "# License"),
        ("# -*- coding: utf-8 -*-\n\n# License", "# License"),
        ("#!/usr/bin/env python3\n# coding: utf-8\n\n# License", "# License"),
    ],
    ids=["no_preamble", "with_shebang", "with_encoding", "with_shebang_and_encoding"],
)
def test_extract_content_after_preamble(content: str, expected: str) -> None:
    assert extract_content_after_preamble(content) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("__pycache__\n.venv\n*.pyc\nbuild/\n", {"__pycache__", ".venv", "*.pyc", "build"}),
        ("# This is a comment\n__pycache__\n# Another comment\n.venv\n", {"__pycache__", ".venv"}),
        ("__pycache__\n\n\n.venv\n", {"__pycache__", ".venv"}),
        ("/build/\n/dist\nvenv/\n", {"build", "dist", "venv"}),
        ("# Python\n__pycache__/\n*.py[cod]\n*.egg-info/\n\n# IDE\n.idea/", {"__pycache__", "*.py[cod]", "*.egg-info", ".idea"}),
    ],
    ids=["standard", "comments", "empty_lines", "slashes", "complex"],
)
def test_parse_gitignore_logic(gitignore_path: Path, tmp_path: Path, content: str, expected: str):
    gitignore_path.write_text(content)

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset(expected)


def test_parse_gitignore_returns_empty_if_no_gitignore(tmp_path: Path):
    """Separate test for the missing file case as it doesn't need the fixture."""
    assert parse_gitignore(tmp_path) == frozenset()


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


@pytest.mark.parametrize(
    ("path", "excludes", "expected"),
    [
        (Path("src/__pycache__/cached.pyc"), frozenset({"__pycache__", ".venv"}), True),
        (Path("src/main.py"), frozenset(), False),
    ],
    ids=["multiple_excludes", "no_excludes"],
)
def test_should_exclude(path: Path, excludes: frozenset[str], expected: bool) -> None:
    assert should_exclude(path, excludes) is expected


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


def test_check_file_header_valid_year(tmp_path: Path) -> None:
    """Test that check_file_header succeeds when the header matches the current year."""
    test_year = 2024
    header = get_license_header(test_year)
    file = tmp_path / "valid_header.py"
    file.write_text(f"{header}\nimport sys\n")

    with patch("scripts.check_license_header.get_current_year", return_value=test_year):
        has_valid, year = check_file_header(file)

    assert has_valid is True
    assert year == test_year


def test_check_file_header_invalid_year(tmp_path: Path) -> None:
    """Test that check_file_header fails when the header year is outdated."""
    header_year = 2020
    current_year = 2024

    header = get_license_header(header_year)
    file = tmp_path / "invalid_header.py"
    file.write_text(f"{header}\nimport sys\n")

    with patch("scripts.check_license_header.get_current_year", return_value=current_year):
        has_valid, year = check_file_header(file)

    assert has_valid is False
    assert year == header_year


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


def test_fix_file_header_no_trailing_newline(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "no_newline.py"
    file.write_text("import sys")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.endswith("\n")
    assert f"Copyright © {current_year}" in content
    assert "import sys" in content


def test_fix_file_header_only_preamble_lines(tmp_path: Path, current_year: int) -> None:
    file = tmp_path / "only_shebang.py"
    file.write_text("#!/usr/bin/env python3\n")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n")
    assert f"Copyright © {current_year}" in content


def test_fix_file_header_preamble_only_no_trailing_newline(tmp_path: Path, current_year: int) -> None:
    """All lines are preamble (no trailing newline), so the for loop exhausts without break."""
    file = tmp_path / "preamble_only.py"
    file.write_text("#!/usr/bin/env python3\n# coding: utf-8")

    result = fix_file_header(file)

    assert result is True
    content = file.read_text()
    assert content.startswith("#!/usr/bin/env python3\n# coding: utf-8\n")
    assert f"Copyright © {current_year}" in content


def test_parse_gitignore_skips_slash_only_lines(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("/\n__pycache__\n")

    patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset({"__pycache__"})


def test_parse_gitignore_handles_oserror(tmp_path: Path) -> None:
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__pycache__\n")

    with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
        patterns = parse_gitignore(tmp_path)

    assert patterns == frozenset()


def test_extract_content_after_preamble_all_lines_are_preamble() -> None:
    content = "#!/usr/bin/env python3\n# coding: utf-8"
    result = extract_content_after_preamble(content)
    assert result == ""


def test_should_exclude_wildcard_prefix_no_match() -> None:
    assert should_exclude(Path("src/main.py"), frozenset({"*.pyc"})) is False


def test_should_exclude_wildcard_prefix_matches_part() -> None:
    assert should_exclude(Path("pkg.egg-info/PKG-INFO"), frozenset({"*.egg-info"})) is True


def test_find_python_files_non_py_file(tmp_path: Path) -> None:
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("hello")

    files = find_python_files([txt_file], frozenset())

    assert files == []


def test_find_python_files_nonexistent_path(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"

    files = find_python_files([nonexistent], frozenset())

    assert files == []


def test_build_excludes_without_gitignore(tmp_path: Path) -> None:
    excludes = build_excludes([tmp_path], [], use_gitignore=False)

    assert ".git" in excludes
    assert ".uv-cache" in excludes


def test_build_excludes_with_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".venv\n__pycache__\n")
    excludes = build_excludes([tmp_path], [], use_gitignore=True)

    assert ".venv" in excludes
    assert "__pycache__" in excludes


def test_build_excludes_with_extra_excludes(tmp_path: Path) -> None:
    excludes = build_excludes([tmp_path], ["custom_dir"], use_gitignore=False)

    assert "custom_dir" in excludes


def test_build_excludes_with_file_path(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text(".venv\n")
    file_path = tmp_path / "main.py"
    file_path.write_text("")

    excludes = build_excludes([file_path], [], use_gitignore=True)

    assert ".venv" in excludes


def test_get_files_to_check_with_files_list(tmp_path: Path) -> None:
    py_file = tmp_path / "main.py"
    py_file.write_text("")
    txt_file = tmp_path / "readme.txt"
    txt_file.write_text("")

    result = get_files_to_check([], [py_file, txt_file], frozenset(), recursive=True)

    assert result == [py_file]


def test_get_files_to_check_with_excluded_file(tmp_path: Path) -> None:
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()
    py_file = cache_dir / "cached.py"
    py_file.write_text("")

    result = get_files_to_check([], [py_file], frozenset({"__pycache__"}), recursive=True)

    assert result == []


def test_get_files_to_check_without_files(project_dir: Path) -> None:
    result = get_files_to_check([project_dir], None, frozenset({"__pycache__", ".venv"}), recursive=True)

    assert len(result) > 0


def test_classify_files_missing_header(tmp_path: Path) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    missing, invalid = classify_files([f])

    assert f in missing
    assert f not in invalid


def test_classify_files_invalid_year(tmp_path: Path) -> None:
    f = tmp_path / "old_year.py"
    f.write_text(f"{get_license_header(2020)}\nimport sys\n")

    missing, invalid = classify_files([f])

    assert f not in missing
    assert f in invalid


def test_classify_files_valid(tmp_path: Path, valid_content: str) -> None:
    f = tmp_path / "valid.py"
    f.write_text(valid_content)

    missing, invalid = classify_files([f])

    assert f not in missing
    assert f not in invalid


def test_classify_files_empty() -> None:
    missing, invalid = classify_files([])

    assert missing == []
    assert invalid == []


def test_apply_fixes_successful(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    apply_fixes([f])

    assert "Fixed:" in capsys.readouterr().out


def test_apply_fixes_failed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "nonexistent.py"

    apply_fixes([f])

    assert "Error:" in capsys.readouterr().out


def test_report_issues_no_issues(capsys: pytest.CaptureFixture[str]) -> None:
    report_issues([], [], verbose=False)

    assert capsys.readouterr().out == ""


def test_report_issues_missing_non_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "missing.py"

    report_issues([f], [], verbose=False)

    out = capsys.readouterr().out
    assert "Missing license header in 1 files" in out
    assert "Use --verbose" in out


def test_report_issues_missing_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "missing.py"

    report_issues([f], [], verbose=True)

    assert str(f) in capsys.readouterr().out


def test_report_issues_invalid_year_non_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "old.py"

    report_issues([], [f], verbose=False)

    out = capsys.readouterr().out
    assert "Invalid year in 1 files" in out
    assert "Use --verbose" in out


def test_report_issues_invalid_year_verbose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "old.py"

    report_issues([], [f], verbose=True)

    assert str(f) in capsys.readouterr().out


def test_create_parser_defaults() -> None:
    parser = create_parser()

    args = parser.parse_args([])

    assert args.fix is False
    assert args.verbose is False
    assert args.no_recursive is False
    assert args.no_gitignore is False
    assert args.exclude == []
    assert args.files is None


def test_create_parser_all_flags() -> None:
    parser = create_parser()

    args = parser.parse_args(["--fix", "--verbose", "--no-recursive", "--no-gitignore"])

    assert args.fix is True
    assert args.verbose is True
    assert args.no_recursive is True
    assert args.no_gitignore is True


def test_create_parser_paths() -> None:
    parser = create_parser()

    args = parser.parse_args(["src/", "tests/"])

    assert len(args.paths) == 2


def test_create_parser_exclude() -> None:
    parser = create_parser()

    args = parser.parse_args(["--exclude", ".venv", "__pycache__"])

    assert ".venv" in args.exclude
    assert "__pycache__" in args.exclude


def test_create_parser_files() -> None:
    parser = create_parser()

    args = parser.parse_args(["--files", "a.py", "b.py"])

    assert len(args.files) == 2


def _make_args(**kwargs: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "paths": [],
        "files": None,
        "exclude": [],
        "no_recursive": False,
        "no_gitignore": True,
        "fix": False,
        "verbose": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_main_all_valid(tmp_path: Path, valid_content: str, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    result = main(_make_args(paths=[tmp_path]))

    assert result == 0
    assert "valid license headers" in capsys.readouterr().out


def test_main_missing_headers(tmp_path: Path) -> None:
    (tmp_path / "no_header.py").write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path]))

    assert result == 1


def test_main_fix_mode(tmp_path: Path) -> None:
    (tmp_path / "no_header.py").write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path], fix=True))

    assert result == 0


def test_main_fix_mode_no_issues(tmp_path: Path, valid_content: str, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    result = main(_make_args(paths=[tmp_path], fix=True))

    assert result == 0
    assert "valid license headers" in capsys.readouterr().out


def test_main_verbose_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    f = tmp_path / "no_header.py"
    f.write_text("import sys\n")

    result = main(_make_args(paths=[tmp_path], verbose=True))

    assert result == 1
    assert str(f) in capsys.readouterr().out


def test_main_parses_argv(tmp_path: Path, valid_content: str) -> None:
    (tmp_path / "valid.py").write_text(valid_content)

    with patch("sys.argv", ["check_license_header.py", str(tmp_path), "--no-gitignore"]):
        result = main()

    assert result == 0
