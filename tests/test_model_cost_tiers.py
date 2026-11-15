"""Prompt-length pricing tiers (B-AI-07): the highest matching input threshold
prices the whole request."""

import pytest

from pi.ai.models import Model, ModelCost, parse_cost_tiers
from pi.ai.types import Usage
from pi.coding_agent.catalog_sync import normalize_catalog
from pi.coding_agent.config import Configuration, write_json
from pi.coding_agent.model_registry import ModelRegistry
from pi.coding_agent.radius_catalog import normalize_gateway_config


def cost_model(**overrides) -> Model:
    rates = {"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25}
    rates.update(overrides)
    return Model(id="m", name="m", api="faux", provider="test", base_url="", costs=ModelCost(**rates))


def usage(**overrides):
    fields = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "cache_write_1h": 0}
    fields.update(overrides)
    return Usage(**fields)


HAIKU_STYLE_TIERS = [{"input_tokens_above": 100000, "input": 0.5, "output": 2.5, "cache_read": 0.05, "cache_write": 0.625}]


# ---------------------------------------------------------------------------
# calculate_cost tier resolution
# ---------------------------------------------------------------------------

def test_at_threshold_uses_base_price():
    result = cost_model(tiers=HAIKU_STYLE_TIERS).calculate_cost(usage(input=100000))
    assert result["input"] == pytest.approx(0.1)  # strictly above the threshold tiers in


def test_above_threshold_prices_the_whole_request():
    result = cost_model(tiers=HAIKU_STYLE_TIERS).calculate_cost(usage(input=100001, output=1000))
    assert result["input"] == pytest.approx(100001 * 0.5 / 1_000_000)
    assert result["output"] == pytest.approx(1000 * 2.5 / 1_000_000)  # not a staircase


def test_highest_matching_tier_wins():
    tiers = [{"input_tokens_above": 100000, "input": 2.0},
             {"input_tokens_above": 200000, "input": 4.0},
             {"input_tokens_above": 50000, "input": 8.0}]
    result = cost_model(tiers=tiers).calculate_cost(usage(input=250000))
    assert result["input"] == pytest.approx(250000 * 4.0 / 1_000_000)


def test_threshold_counts_cache_read_and_cache_write():
    tiers = HAIKU_STYLE_TIERS
    below = cost_model(tiers=tiers).calculate_cost(usage(input=60000, cache_read=40000))
    assert below["input"] == pytest.approx(60000 * 1.0 / 1_000_000)
    above = cost_model(tiers=tiers).calculate_cost(usage(input=60000, cache_read=40001, cache_write=10))
    assert above["input"] == pytest.approx(60000 * 0.5 / 1_000_000)
    assert above["cache_read"] == pytest.approx(40001 * 0.05 / 1_000_000)


def test_missing_tier_rates_keep_base_price():
    result = cost_model(tiers=[{"input_tokens_above": 100000, "input": 5.0}]).calculate_cost(
        usage(input=150000, output=1000))
    assert result["input"] == pytest.approx(150000 * 5.0 / 1_000_000)
    assert result["output"] == pytest.approx(1000 * 5.0 / 1_000_000)  # base output rate


def test_cache_write_1h_bills_at_twice_the_effective_input_rate():
    result = cost_model(tiers=HAIKU_STYLE_TIERS).calculate_cost(usage(input=150000, cache_write_1h=1000))
    assert result["cache_write_1h"] == pytest.approx(1000 * 2 * 0.5 / 1_000_000)


def test_dedicated_1h_price_survives_tiers():
    model = cost_model(cache_write_1h=3.0, tiers=HAIKU_STYLE_TIERS)
    result = model.calculate_cost(usage(input=150000, cache_write_1h=1000))
    assert result["cache_write_1h"] == pytest.approx(1000 * 3.0 / 1_000_000)


def test_malformed_tier_entries_are_ignored():
    tiers = ["bad", {"input_tokens_above": "x"}, {"tier": {"type": "window", "size": 10}},
             {"input_tokens_above": 100000, "input": 5.0}]
    result = cost_model(tiers=tiers).calculate_cost(usage(input=150000))
    assert result["input"] == pytest.approx(150000 * 5.0 / 1_000_000)


def test_no_tiers_matches_flat_pricing():
    plain = usage(input=123, output=7, cache_read=11, cache_write=13)
    assert cost_model().calculate_cost(plain) == cost_model(tiers=[]).calculate_cost(plain)


# ---------------------------------------------------------------------------
# parse_cost_tiers: catalog wire shapes
# ---------------------------------------------------------------------------

def test_parse_cost_tiers_models_dev_shape():
    raw = [
        {"input": 2, "tier": {"type": "context", "size": 50000}},          # partial: base fills the rest
        {"tier": {"type": "window", "size": 10}},                          # non-context: dropped
        {"input_tokens_above": 100000, "input": 3, "cacheRead": 0.2},      # normalized + camelCase rates
        {"tier": {"type": "context", "size": -1}},                         # invalid threshold: dropped
        "bad",
    ]
    tiers = parse_cost_tiers(raw, base={"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25})
    assert tiers == [
        {"input_tokens_above": 50000, "input": 2, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25},
        {"input_tokens_above": 100000, "input": 3, "output": 5.0, "cache_read": 0.2, "cache_write": 1.25},
    ]


def test_parse_cost_tiers_gateway_shape():
    tiers = parse_cost_tiers([{"inputTokensAbove": 100000, "input": 4, "cacheRead": 0.2}],
                             base={"input": 1.0, "output": 5.0, "cache_read": 0.1, "cache_write": 1.25})
    assert tiers == [{"input_tokens_above": 100000, "input": 4, "output": 5.0, "cache_read": 0.2, "cache_write": 1.25}]


def test_parse_cost_tiers_absent_or_empty():
    assert parse_cost_tiers(None) is None
    assert parse_cost_tiers([]) is None
    assert parse_cost_tiers([{}]) is None
    assert parse_cost_tiers("tiers") is None


# ---------------------------------------------------------------------------
# Catalog ingestion keeps tiers
# ---------------------------------------------------------------------------

def test_catalog_sync_keeps_models_dev_tiers():
    data = {"openai": {"models": {"long-context": {
        "name": "Long Context", "reasoning": False, "input": ["text"],
        "limit": {"context": 200000, "output": 8000},
        "cost": {"input": 1, "output": 2, "cache_read": 0.1, "cache_write": 0.2,
                 "tiers": [{"input": 2, "output": 6, "tier": {"type": "context", "size": 100000}}]},
    }}}}
    entry = normalize_catalog(data)["openai"]["models"][0]
    assert entry["costs"]["tiers"] == [
        {"input_tokens_above": 100000, "input": 2, "output": 6, "cache_read": 0.1, "cache_write": 0.2}]


def test_radius_gateway_cost_tiers():
    config = normalize_gateway_config({"baseUrl": "https://gw.example/v1", "models": [
        {"id": "m", "name": "M", "reasoning": False, "input": ["text"],
         "cost": {"input": 1, "output": 2, "cacheRead": 0.1, "cacheWrite": 0.2,
                  "tiers": [{"inputTokensAbove": 100000, "input": 4}]},
         "contextWindow": 200000, "maxTokens": 8000}]}, "https://gw.example")
    assert config["models"][0]["costs"]["tiers"] == [
        {"input_tokens_above": 100000, "input": 4, "output": 2, "cache_read": 0.1, "cache_write": 0.2}]


# ---------------------------------------------------------------------------
# models.json custom models
# ---------------------------------------------------------------------------

def write_registry(tmp_path, costs):
    agent = tmp_path / "agent"
    write_json(agent / "models.json", {"providers": {"openai": {"base_url": "https://api.openai.com/v1", "models": [
        {"id": "tiered", "context_window": 100000, "costs": costs}]}}})
    return ModelRegistry(Configuration(tmp_path, agent))


def test_models_json_costs_accept_tiers(tmp_path):
    registry = write_registry(tmp_path, {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25,
                                         "tiers": [{"input_tokens_above": 100000, "input": 5, "output": 25}]})
    model = registry.models["openai/tiered"]
    result = model.calculate_cost(usage(input=200000, output=1000))
    assert result["input"] == pytest.approx(200000 * 5 / 1_000_000)
    assert result["output"] == pytest.approx(1000 * 25 / 1_000_000)


def test_models_json_rejects_malformed_tiers(tmp_path):
    with pytest.raises(Exception, match="costs"):
        write_registry(tmp_path, {"input": 1, "tiers": "bad"})
    with pytest.raises(Exception, match="costs"):
        write_registry(tmp_path, {"input": 1, "tiers": [{}]})
