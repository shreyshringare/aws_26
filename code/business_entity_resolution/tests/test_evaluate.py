"""Tests for the exact challenge metric and blocking report."""
import pytest

from src.evaluate import blocking_report, breakdown, f05_entity, score


def test_problem_statement_example():
    assert f05_entity({"a", "b", "x"}, {"a", "b"}) == pytest.approx(0.7142857, abs=1e-6)


def test_singleton_rules():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"x"}, set()) == 0.0
    assert f05_entity(set(), {"a"}) == 0.0
    assert f05_entity({"x"}, {"a"}) == 0.0


def test_macro_average_and_breakdown():
    gt = {"s1": {"a"}, "s2": set()}
    pred = {"s1": ["a"], "s2": ["z"]}
    mean, vals = score(pred, gt)
    assert mean == 0.5 and list(vals) == [1.0, 0.0]
    bd = breakdown(pred, gt, ["s1", "s2"], ["US", "France"])
    assert bd["country=US"] == 1.0 and bd["country=France"] == 0.0
    assert bd["singletons"] == 0.0 and bd["matched"] == 1.0


def test_blocking_report():
    gt = {"s1": {"a", "b"}, "s2": set()}
    rep = blocking_report({"s1": ["a", "c"], "s2": ["d"]}, gt, ["s1", "s2"])
    assert rep["pair_recall"] == 0.5
    assert rep["entity_full_recall"] == 0.5
    assert rep["cands_per_entity"] == 1.5
