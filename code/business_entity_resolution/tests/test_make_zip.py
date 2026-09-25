"""Tests for scripts/make_zip.py: submission zip layout, exclusions, and missing-input errors."""
import zipfile

import pytest

from scripts import make_zip


def _write_pkg_tree(pkg, with_junk=True):
    """Create a minimal (optionally junk-polluted) code/business_entity_resolution/ tree under pkg."""
    src = pkg / "src"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "config.py").write_text("SEED = 42\n", encoding="utf-8")
    sub = src / "subpkg"
    sub.mkdir()
    (sub / "helper.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (pkg / "README.md").write_text("# README\n", encoding="utf-8")
    (pkg / "requirements.txt").write_text("lightgbm==4.0.0\n", encoding="utf-8")
    if with_junk:
        (src / "__pycache__").mkdir()
        (src / "__pycache__" / "config.cpython-311.pyc").write_bytes(b"\x00")
        cache = pkg / "cache"
        cache.mkdir()
        (cache / "big_blob.parquet").write_bytes(b"\x00" * 10)
        tests_dir = pkg / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_config.py").write_text("def test_x(): assert True\n", encoding="utf-8")
        notebooks = pkg / "notebooks"
        notebooks.mkdir()
        (notebooks / "dashboard.ipynb").write_text("{}", encoding="utf-8")
        venv = pkg / ".venv"
        venv.mkdir()
        (venv / "marker.txt").write_text("venv", encoding="utf-8")


def _write_output(root, with_files=True):
    """Create student_resource/output/ under root, optionally with the two required TSVs."""
    out = root / "output"
    out.mkdir()
    if with_files:
        (out / "matching_results.tsv").write_text(
            "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n", encoding="utf-8")
        (out / "candidate_pairs.tsv").write_text(
            "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-2\n", encoding="utf-8")


def _full_tree(tmp_path, with_junk=True, with_output=True, with_doc=True):
    """Build a full fake student_resource/ tree (root, pkg) ready for build_zip()."""
    root = tmp_path / "student_resource"
    pkg = root / "code" / "business_entity_resolution"
    _write_pkg_tree(pkg, with_junk=with_junk)
    _write_output(root, with_files=with_output)
    if with_doc:
        (root / "Documentation_template.md").write_text("# Methodology\n", encoding="utf-8")
    return root, pkg


def test_check_required_inputs_passes_when_everything_present(tmp_path):
    """No exception when output TSVs, README, requirements.txt, src/, and the doc all exist."""
    root, pkg = _full_tree(tmp_path)
    make_zip.check_required_inputs(root=root, pkg=pkg)


def test_check_required_inputs_missing_output_raises_actionable_error(tmp_path):
    """Missing output/ TSVs raise FileNotFoundError naming the exact missing paths."""
    root, pkg = _full_tree(tmp_path, with_output=False)
    with pytest.raises(FileNotFoundError) as exc_info:
        make_zip.check_required_inputs(root=root, pkg=pkg)
    msg = str(exc_info.value)
    assert "matching_results.tsv" in msg
    assert "candidate_pairs.tsv" in msg
    assert "run_pipeline" in msg  # actionable: tells the user how to fix it


def test_check_required_inputs_missing_doc_raises(tmp_path):
    """Missing Documentation_template.md is also caught with a clear message."""
    root, pkg = _full_tree(tmp_path, with_doc=False)
    with pytest.raises(FileNotFoundError) as exc_info:
        make_zip.check_required_inputs(root=root, pkg=pkg)
    assert "Documentation_template.md" in str(exc_info.value)


def test_build_zip_has_exact_required_layout(tmp_path):
    """The built zip contains exactly the paths required by the problem statement, no more."""
    root, pkg = _full_tree(tmp_path, with_junk=False)
    zip_path = make_zip.build_zip("acme", root=root, pkg=pkg)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert names == {
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "code/business_entity_resolution/src/__init__.py",
        "code/business_entity_resolution/src/config.py",
        "code/business_entity_resolution/src/subpkg/helper.py",
        "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt",
        "Documentation_template.md",
    }


def test_build_zip_excludes_cache_pycache_tests_notebooks_venv(tmp_path):
    """cache/, __pycache__/, tests/, notebooks/, and .venv/ never end up in the zip."""
    root, pkg = _full_tree(tmp_path, with_junk=True)
    zip_path = make_zip.build_zip("acme", root=root, pkg=pkg)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    for bad in ("cache/", "__pycache__", ".pyc", "/tests/", "notebooks/", ".venv/", ".ipynb"):
        assert not any(bad in n for n in names), f"{bad!r} leaked into zip: {names}"


def test_verify_zip_matches_expected_members(tmp_path):
    """verify_zip() succeeds and returns a sorted listing matching expected_members()."""
    root, pkg = _full_tree(tmp_path, with_junk=False)
    zip_path = make_zip.build_zip("acme", root=root, pkg=pkg)
    listing = make_zip.verify_zip(zip_path, root=root, pkg=pkg)
    assert listing == sorted(listing)
    assert set(listing) == make_zip.expected_members(root=root, pkg=pkg)


def test_verify_zip_raises_on_mismatch(tmp_path):
    """verify_zip() raises AssertionError if the zip contains something unexpected."""
    root, pkg = _full_tree(tmp_path, with_junk=False)
    zip_path = make_zip.build_zip("acme", root=root, pkg=pkg)
    with zipfile.ZipFile(zip_path, "a") as zf:
        zf.writestr("stray_extra_file.txt", "oops")
    with pytest.raises(AssertionError):
        make_zip.verify_zip(zip_path, root=root, pkg=pkg)


def test_zip_filename_uses_team_argument(tmp_path):
    """build_zip() names the archive <team>_submission.zip under root."""
    root, pkg = _full_tree(tmp_path, with_junk=False)
    zip_path = make_zip.build_zip("gstack_wizards", root=root, pkg=pkg)
    assert zip_path == root / "gstack_wizards_submission.zip"
    assert zip_path.is_file()


def test_main_exits_with_error_when_output_missing(tmp_path, monkeypatch, capsys):
    """main() exits non-zero with a stderr message (not a broken/incomplete zip) when inputs are missing."""
    root, pkg = _full_tree(tmp_path, with_output=False)
    monkeypatch.setattr(make_zip, "ROOT", root)
    monkeypatch.setattr(make_zip, "PKG", pkg)
    with pytest.raises(SystemExit) as exc_info:
        make_zip.main(["--team", "acme"])
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "matching_results.tsv" in err
    assert not (root / "acme_submission.zip").exists()


def test_main_builds_and_lists_zip_when_inputs_present(tmp_path, monkeypatch, capsys):
    """main() builds the zip and prints its member listing plus the output path/size line."""
    root, pkg = _full_tree(tmp_path, with_junk=False)
    monkeypatch.setattr(make_zip, "ROOT", root)
    monkeypatch.setattr(make_zip, "PKG", pkg)
    make_zip.main(["--team", "acme"])
    out = capsys.readouterr().out
    assert "output/matching_results.tsv" in out
    assert "code/business_entity_resolution/README.md" in out
    assert "Documentation_template.md" in out
    assert "acme_submission.zip" in out
    assert (root / "acme_submission.zip").is_file()
