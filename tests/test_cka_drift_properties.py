"""Property-based tests for cka_drift.py.

Feature: cka-representation-drift
Tests use hypothesis with @settings(max_examples=100, deadline=None).
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from src.utils.cka_drift import ARCHITECTURE_LAYER_MAP


# ---------------------------------------------------------------------------
# Property 1: Architecture Layer Map Completeness
# Feature: cka-representation-drift, Property 1: Architecture Layer Map Completeness
# Validates: Requirements 2.1, 2.7
# ---------------------------------------------------------------------------

@given(st.sampled_from(list(ARCHITECTURE_LAYER_MAP.keys())))
@settings(max_examples=100, deadline=None)
def test_architecture_layer_map_completeness(key):
    """For any architecture key in ARCHITECTURE_LAYER_MAP, the value is a
    non-empty list of strings.

    # Feature: cka-representation-drift, Property 1: Architecture Layer Map Completeness
    """
    layer_spec = ARCHITECTURE_LAYER_MAP[key]

    # Must be a list
    assert isinstance(layer_spec, list), (
        f"ARCHITECTURE_LAYER_MAP[{key!r}] should be a list, got {type(layer_spec)}"
    )

    # Must be non-empty
    assert len(layer_spec) > 0, (
        f"ARCHITECTURE_LAYER_MAP[{key!r}] should be non-empty"
    )

    # Every element must be a string
    for i, entry in enumerate(layer_spec):
        assert isinstance(entry, str), (
            f"ARCHITECTURE_LAYER_MAP[{key!r}][{i}] should be a str, got {type(entry)}"
        )
