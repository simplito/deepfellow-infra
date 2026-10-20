# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""Tests for check_example_env.py module."""

from pathlib import Path

from pydantic import AliasChoices, AliasPath, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typer.testing import CliRunner

from scripts.check_example_env import (
    app,
    expected_names_for_field,
    find_missing,
    get_expected_env_names,
    parse_existing_keys,
)
from server.config import AppSettings
from server.dynamic_config import DynamicSettings

runner = CliRunner()

BOOTSTRAP_FIELD_NAMES = {
    "infra_admin_api_key",
    "docker_subnet",
    "external_only",
    "storage_dir",
    "storage_services_dir",
    "container_name_prefix",
    "compose_prefix",
    "docker_hub_token",
    "ollama_vram_overhead_factor",
}


class PlainField(BaseModel):
    """A model with a single field carrying no validation_alias."""

    name: str = "x"


class StrAliasField(BaseModel):
    """A model with a field aliased to a raw string name."""

    name: str = Field(default="x", validation_alias="RAW_ALIAS")


class ChoicesAliasField(BaseModel):
    """A model with a field aliased via AliasChoices."""

    name: str = Field(default="x", validation_alias=AliasChoices("FIRST_CHOICE", "SECOND_CHOICE"))


class PathAliasField(BaseModel):
    """A model with a field aliased via AliasPath - not a top-level env name."""

    name: str = Field(default="x", validation_alias=AliasPath("nested", 0))


class DummySettings(BaseSettings):
    """A small BaseSettings-shaped fixture standing in for AppSettings/DynamicSettings."""

    model_config = SettingsConfigDict(env_prefix="DF_")

    plain_field: str = ""
    aliased_field: str = Field(default="", validation_alias=AliasChoices("RAW_ALIAS_NAME"))
    dynamic_field: str = ""


def test_expected_names_for_field_no_alias_uses_prefixed_upper_name() -> None:
    field_info = PlainField.model_fields["name"]

    assert expected_names_for_field("name", field_info, "DF_") == {"DF_NAME"}


def test_expected_names_for_field_str_alias_bypasses_prefix() -> None:
    field_info = StrAliasField.model_fields["name"]

    assert expected_names_for_field("name", field_info, "DF_") == {"RAW_ALIAS"}


def test_expected_names_for_field_alias_choices_bypasses_prefix() -> None:
    field_info = ChoicesAliasField.model_fields["name"]

    assert expected_names_for_field("name", field_info, "DF_") == {"FIRST_CHOICE", "SECOND_CHOICE"}


def test_expected_names_for_field_alias_path_yields_no_names() -> None:
    field_info = PathAliasField.model_fields["name"]

    assert expected_names_for_field("name", field_info, "DF_") == set()


def test_get_expected_env_names_excludes_given_dynamic_field_names() -> None:
    names = get_expected_env_names(DummySettings, dynamic_field_names={"dynamic_field"})

    assert names == {"DF_PLAIN_FIELD", "RAW_ALIAS_NAME"}


def test_bootstrap_field_names_are_exactly_app_settings_minus_dynamic_settings() -> None:
    assert set(AppSettings.model_fields) - set(DynamicSettings.model_fields) == BOOTSTRAP_FIELD_NAMES


def test_get_expected_env_names_against_real_app_settings() -> None:
    names = get_expected_env_names(AppSettings)
    expected_bootstrap_names = {
        "DF_INFRA_ADMIN_API_KEY",
        "DF_DOCKER_SUBNET",
        "DF_EXTERNAL_ONLY",
        "DF_STORAGE_DIR",
        "DF_STORAGE_SERVICES_DIR",
        "DF_CONTAINER_NAME_PREFIX",
        "DF_COMPOSE_PREFIX",
        "DF_DOCKER_HUB_TOKEN",
        "OLLAMA_VRAM_OVERHEAD_FACTOR",
    }

    assert names == expected_bootstrap_names


def test_get_expected_env_names_excludes_every_dynamic_settings_field() -> None:
    names = get_expected_env_names(AppSettings)
    dynamic_env_names = {f"DF_{field_name.upper()}" for field_name in DynamicSettings.model_fields}

    assert names.isdisjoint(dynamic_env_names)


def test_parse_existing_keys_ignores_comments_and_values() -> None:
    content = "\n".join(
        [
            "DF_FOO=bar",
            "# a comment",
            "# DF_COMMENTED_OUT=1",
            "OLLAMA_VRAM_OVERHEAD_FACTOR=1.0",
            "",
        ]
    )

    assert parse_existing_keys(content) == {"DF_FOO", "OLLAMA_VRAM_OVERHEAD_FACTOR"}


def test_parse_existing_keys_tolerates_whitespace_around_equals_and_key() -> None:
    content = "\n".join(
        [
            "DF_FOO = bar",
            "  DF_INDENTED = xyz",
            "DF_TAB\t=\ttabbed",
            "  # still a comment despite leading whitespace",
        ]
    )

    assert parse_existing_keys(content) == {"DF_FOO", "DF_INDENTED", "DF_TAB"}


def test_find_missing_ignores_case_of_existing_keys_when_case_insensitive() -> None:
    assert find_missing({"DF_FOO"}, {"df_foo"}, case_sensitive=False) == []


def test_find_missing_reports_truly_absent_names_sorted() -> None:
    assert find_missing({"DF_B", "DF_A", "DF_C"}, {"DF_B"}, case_sensitive=False) == ["DF_A", "DF_C"]


def test_find_missing_requires_exact_case_when_case_sensitive() -> None:
    assert find_missing({"DF_FOO"}, {"df_foo"}, case_sensitive=True) == ["DF_FOO"]


def test_find_missing_accepts_exact_case_match_when_case_sensitive() -> None:
    assert find_missing({"DF_FOO"}, {"DF_FOO"}, case_sensitive=True) == []


def test_cli_exits_nonzero_for_a_nonexistent_path(tmp_path: Path) -> None:
    result = runner.invoke(app, [str(tmp_path / "does-not-exist.env")])

    assert result.exit_code != 0


def test_cli_ignores_a_leading_bom_when_matching_the_first_line(tmp_path: Path) -> None:
    expected = get_expected_env_names(AppSettings)
    env_file = tmp_path / "example.env"
    content = "\n".join(f"{name}=x" for name in sorted(expected))
    env_file.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

    result = runner.invoke(app, [str(env_file)])

    assert result.exit_code == 0


def test_cli_exits_zero_when_every_expected_variable_is_present(tmp_path: Path) -> None:
    expected = get_expected_env_names(AppSettings)
    env_file = tmp_path / "example.env"
    original_content = "\n".join(f"{name}=x" for name in expected)
    env_file.write_text(original_content)
    original_mtime = env_file.stat().st_mtime_ns

    result = runner.invoke(app, [str(env_file)])

    assert result.exit_code == 0
    assert "all bootstrap AppSettings variables are documented" in result.stdout
    assert env_file.read_text() == original_content
    assert env_file.stat().st_mtime_ns == original_mtime


def test_cli_exits_nonzero_and_reports_missing_variable(tmp_path: Path) -> None:
    expected = get_expected_env_names(AppSettings)
    missing_name = sorted(expected)[0]
    remaining = expected - {missing_name}
    env_file = tmp_path / "example.env"
    original_content = "\n".join(f"{name}=x" for name in remaining)
    env_file.write_text(original_content)
    original_mtime = env_file.stat().st_mtime_ns

    result = runner.invoke(app, [str(env_file)])

    assert result.exit_code == 1
    assert missing_name in result.stdout
    assert env_file.read_text() == original_content
    assert env_file.stat().st_mtime_ns == original_mtime
