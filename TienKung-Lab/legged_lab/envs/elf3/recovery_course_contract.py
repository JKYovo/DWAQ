"""Only a relocated pose-bank path may differ in an init-v4 checkpoint contract."""
from copy import deepcopy


def validate_portable_contract(saved, expected):
    if (saved.get("infos") or {}).get("initialization_only"):
        raise ValueError("An initialization snapshot cannot resume training")
    actual = (saved.get("infos") or {}).get("elf3_contract")
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("Missing recovery checkpoint contract")
    normalized = []
    for contract in (actual, expected):
        contract = deepcopy(contract)
        try:
            bank = contract["recovery_initialization_experiment"]["bank"]
            if not isinstance(bank["path"], str) or not bank["path"]:
                raise ValueError("Missing initialization bank location")
            del bank["path"]
        except (KeyError, TypeError) as error:
            raise ValueError("Require the complete init-v4 pose-bank identity") from error
        normalized.append(contract)
    if normalized[0] != normalized[1]:
        raise ValueError("Recovery recipe, physics, motion data or source hash differs; only bank location may change")
