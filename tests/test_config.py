"""Tests for configuration management, CLI argument parsing, and catalog loading."""

import json
import os
import tempfile
import pytest
from unittest.mock import patch

from run import load_catalog


class TestLoadCatalog:
    """Tests for catalog loading in run.py"""

    def test_load_default_copilot(self):
        result = load_catalog(catalog_dir="copilot", catalog_name="default_copilot")
        assert isinstance(result, dict)
        assert "learning_objectives" in result
        assert "syllabus" in result
        assert "slides" in result
        assert "script" in result
        assert "assessment" in result
        assert "overall" in result
        # All values should be empty strings for defaults
        for v in result.values():
            assert v == ""

    def test_load_catalog_from_file(self):
        catalog_data = {
            "course_structure": {"topics": ["ML", "DL"]},
            "student_profile": {"level": "graduate"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = os.path.join(tmpdir, "test_catalog.json")
            with open(catalog_path, "w") as f:
                json.dump(catalog_data, f)

            result = load_catalog(catalog_dir=tmpdir, catalog_name="test_catalog")

        assert result["course_structure"]["topics"] == ["ML", "DL"]
        assert result["student_profile"]["level"] == "graduate"

    def test_load_catalog_missing_file(self):
        result = load_catalog(catalog_dir="/nonexistent", catalog_name="missing")
        assert result == {}

    def test_load_catalog_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = os.path.join(tmpdir, "bad.json")
            with open(bad_file, "w") as f:
                f.write("not valid json {{{")

            result = load_catalog(catalog_dir=tmpdir, catalog_name="bad")
        assert result == {}


class TestCLIArgs:
    """Tests for CLI argument parsing structure."""

    def test_run_py_importable(self):
        """run.py should be importable without side effects."""
        import run
        assert hasattr(run, "run_instructional_design")
        assert hasattr(run, "run_optimization")
        assert hasattr(run, "load_catalog")
