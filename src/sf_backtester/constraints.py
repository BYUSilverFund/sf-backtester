"""Shared constraint specifications for configurations and worker processes."""

import argparse
import json

import sf_quant.optimizer as sfo


ConstraintSpec = str | dict[str, object]

CONSTRAINT_REGISTRY: dict[str, type] = {
    "ZeroBeta": sfo.ZeroBeta,
    "ZeroInvestment": sfo.ZeroInvestment,
    "UnitBeta": sfo.UnitBeta,
    "FullInvestment": sfo.FullInvestment,
    "LongOnly": sfo.LongOnly,
    "NoBuyingOnMargin": sfo.NoBuyingOnMargin,
    "GrossLeverage": sfo.GrossLeverage,
}


def get_constraints(specs: list[ConstraintSpec]) -> list[sfo.Constraint]:
    """Validate names or flat named parameter mappings and construct constraints.

    Existing names such as ``"LongOnly"`` retain their default behavior.
    Parameterized specifications use ``{"name": "GrossLeverage",
    "max_leverage": 1.6}``. Only registered constraints and their supported
    parameters are accepted; leverage validation belongs to ``GrossLeverage``.
    """
    if not isinstance(specs, list):
        raise TypeError("constraints must be a list of names or named mappings.")

    constraints = []
    for index, spec in enumerate(specs):
        if isinstance(spec, str):
            name = spec
            parameters = {}
        elif isinstance(spec, dict):
            name = spec.get("name")
            if not isinstance(name, str):
                raise ValueError(f"Constraint {index} must have a string 'name'.")
            parameters = {key: value for key, value in spec.items() if key != "name"}
        else:
            raise TypeError(f"Constraint {index} must be a name or a named mapping.")

        if name not in CONSTRAINT_REGISTRY:
            available = ", ".join(CONSTRAINT_REGISTRY)
            raise KeyError(f"Unknown constraint '{name}'. Available: {available}")

        allowed = {"max_leverage"} if name == "GrossLeverage" else set()
        unsupported = parameters.keys() - allowed
        if unsupported:
            fields = ", ".join(repr(key) for key in unsupported)
            raise ValueError(f"Unsupported parameters for {name}: {fields}")

        constraints.append(CONSTRAINT_REGISTRY[name](**parameters))

    return constraints


def serialize_constraints(specs: list[ConstraintSpec]) -> str:
    """Validate specifications and encode them as a single JSON argument."""
    constraints = get_constraints(specs)
    normalized = []
    for spec, constraint in zip(specs, constraints):
        if isinstance(spec, dict):
            spec = dict(spec)
            if spec["name"] == "GrossLeverage" and "max_leverage" in spec:
                spec["max_leverage"] = constraint.max_leverage
        normalized.append(spec)
    return json.dumps(normalized, allow_nan=False)


def _parse_constraints_json(value: str) -> list[ConstraintSpec]:
    try:
        specs = json.loads(value)
        get_constraints(specs)
    except (KeyError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return specs


def add_constraint_arguments(parser: argparse.ArgumentParser) -> None:
    """Accept JSON specifications or the legacy list of constraint names."""
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--constraints-json",
        type=_parse_constraints_json,
        help="JSON array of names or named parameter mappings; use '[]' for none.",
    )
    group.add_argument(
        "--constraints",
        nargs="*",
        help="Legacy constraint names (e.g., ZeroBeta ZeroInvestment); may be empty.",
    )


def constraints_from_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser | None = None,
) -> list[ConstraintSpec]:
    """Return validated worker specifications while preserving an empty list."""
    specs = args.constraints_json if args.constraints_json is not None else args.constraints
    try:
        get_constraints(specs)
    except (KeyError, TypeError, ValueError) as exc:
        if parser is not None:
            parser.error(str(exc))
        raise
    return specs
