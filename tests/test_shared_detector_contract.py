from __future__ import annotations

from scripts.validate_shared_detector import validate_contract


def _contract():
    return {
        "schema": "manga-translator.shared-detector-contract.v1",
        "input": {"name": "images", "dtype": "float32", "shape": [1, 3, 1024, 1024]},
        "heads": {
            name: {
                "output": f"{name}_head",
                "task": task,
                "destructive_authority": False,
                "requires_mask_gate": True,
            }
            for name, task in {"bubble": "proposal", "text": "segmentation", "stroke": "segmentation"}.items()
        },
        "authority_policy": "existing-text-segmenter-mask-gate",
    }


def test_shared_detector_contract_accepts_three_guarded_heads():
    assert validate_contract(_contract()) == []


def test_shared_detector_contract_rejects_direct_authority():
    contract = _contract()
    contract["heads"]["text"]["destructive_authority"] = True
    errors = validate_contract(contract)
    assert any("cannot declare destructive authority" in error for error in errors)
