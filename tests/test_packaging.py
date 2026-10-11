"""Verification tests for submission packaging and integrity."""
import os
import zipfile
import json
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_bastesh_manifest_and_zip():
    from rubbish import cmd_bastesh
    import argparse
    args = argparse.Namespace(key="test-pack-key")
    cmd_bastesh(args)

    zip_path = os.path.join(BASE, "submissions", "submission.zip")
    assert os.path.exists(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        assert "manifest.json" in names
        assert any(n.startswith("rubbish_rag/") for n in names)
        # Check manifest content
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        assert manifest["key"] == "test-pack-key"
