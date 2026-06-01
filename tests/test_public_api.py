"""The public API surface is a deliberate contract (see consult/__init__.py).

This golden test fails when `consult.__all__` drifts, so any intended change
to the supported surface has to be made here in the same commit. An accidental
export or removal then can't slip into a release unnoticed.
"""

from __future__ import annotations

import consult

EXPECTED_PUBLIC_API = {
    "__version__",
    # Functions
    "consult",
    "panel",
    "synthesise",
    # Pydantic models
    "AnyCapsule",
    "ArbiterVerdict",
    "Capsule",
    "Finding",
    "ManifestEntry",
    "ModelSpec",
    "RefineResult",
    "ResearchCapsule",
    "ReviewCapsule",
    "RunHandle",
    "RunResult",
    "SequenceResult",
    "Status",
    # Exception taxonomy
    "ConsultError",
    "UnknownModelError",
    "BudgetExceededError",
    "PathTrustError",
    "ProviderError",
    "CapsuleParseError",
}


def test_public_api_surface_matches_golden():
    assert set(consult.__all__) == EXPECTED_PUBLIC_API


def test_everything_in_all_is_importable():
    for name in consult.__all__:
        assert hasattr(consult, name), f"{name} is in __all__ but not importable from consult"
