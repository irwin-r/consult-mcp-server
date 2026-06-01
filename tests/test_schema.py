"""The shipped capsule JSON Schema (`consult/schemas/capsule_v2.json`) is a
published reference for consumers. It is hand-maintained rather than generated,
so this test keeps it consistent with the Pydantic models it documents — if a
finding severity / category or an overall-verdict value drifts in the models,
the schema file has to move with it.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

import consult
from consult.types import Finding, ReviewCapsule

SCHEMA_PATH = Path(consult.__file__).parent / "schemas" / "capsule_v2.json"


def _literal_values(model, field):
    return set(typing.get_args(model.model_fields[field].annotation))


def test_capsule_schema_is_well_formed():
    schema = json.loads(SCHEMA_PATH.read_text())
    titles = {variant["title"] for variant in schema["oneOf"]}
    assert titles == {"DecisionCapsule", "ReviewCapsule", "ResearchCapsule"}


def test_capsule_schema_review_enums_match_models():
    schema = json.loads(SCHEMA_PATH.read_text())
    review = next(v for v in schema["oneOf"] if v["title"] == "ReviewCapsule")
    finding_props = review["properties"]["findings"]["items"]["properties"]

    assert set(finding_props["severity"]["enum"]) == _literal_values(Finding, "severity")
    assert set(finding_props["category"]["enum"]) == _literal_values(Finding, "category")
    assert set(review["properties"]["overall_verdict"]["enum"]) == _literal_values(
        ReviewCapsule, "overall_verdict"
    )
