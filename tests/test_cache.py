"""Persistent cache location and uncleaned lineage entries (0.9.0)."""

import os
import pickle

import pytest

import taxodist.fetch as fetch


def test_default_cache_file_honours_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TAXODIST_CACHE_DIR", str(tmp_path))
    assert fetch._default_cache_file() == os.path.join(str(tmp_path), "lineage-cache.pkl")


@pytest.mark.parametrize(
    "platform, env, expected_dir",
    [
        ("win32", {"LOCALAPPDATA": "C:/Local"}, "C:/Local/taxodist/Cache"),
        ("win32", {}, "/taxodist/Cache"),
        ("darwin", {}, "/Library/Caches/taxodist"),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, "/xdg/taxodist"),
        ("linux", {}, "/.cache/taxodist"),
    ],
)
def test_default_cache_file_per_platform(monkeypatch, platform, env, expected_dir):
    for name in ("TAXODIST_CACHE_DIR", "LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(fetch.sys, "platform", platform)
    path = fetch._default_cache_file().replace("\\", "/")
    assert path.endswith(expected_dir + "/lineage-cache.pkl")


def test_save_and_load_default_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "nested" / "cache"
    monkeypatch.setenv("TAXODIST_CACHE_DIR", str(cache_dir))
    fetch.clear_cache()
    fetch._taxodist_cache["lin_1"] = ["Biota", "Alpha"]
    path = fetch.save_cache()
    assert os.path.exists(path)
    fetch.clear_cache()
    fetch.load_cache()
    assert fetch._taxodist_cache["lin_1"] == ["Biota", "Alpha"]
    fetch.clear_cache()


def test_load_cache_validates_uncleaned_lineages(tmp_path):
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as stream:
        pickle.dump({"rawlin_1": 1}, stream)
    with pytest.raises(ValueError, match="lineage entries"):
        fetch.load_cache(str(path))
