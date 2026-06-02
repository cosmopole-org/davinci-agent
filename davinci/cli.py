"""Minimal CLI for driving Davinci locally.

Usage:
    python -m davinci.cli "<objective>" [--mode auto|plan|default] [--json]
    python -m davinci.cli --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .caspar_runtime import _build_registry, _capability_snapshot
from .engine import DavinciAgent, EchoExecutor
from .observability import Budget, Tracer
from .permissions import PermissionEngine, PermissionMode, Risk


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="davinci", description="Davinci agent CLI")
    parser.add_argument("objective", nargs="?", default="Run a self-test and report capabilities.")
    parser.add_argument("--mode", default="auto",
                        choices=[m.value for m in PermissionMode])
    parser.add_argument("--categories", default="", help="comma-separated required categories")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--self-test", action="store_true", help="print capability snapshot and exit")
    args = parser.parse_args(argv)

    registry = _build_registry()
    if args.self_test:
        print(json.dumps(_capability_snapshot(registry), indent=2))
        return 0

    required = [c.strip() for c in args.categories.split(",") if c.strip()]
    agent = DavinciAgent(
        registry=registry,
        permissions=PermissionEngine(mode=PermissionMode(args.mode), risk_ceiling=Risk.MEDIUM),
        executor=EchoExecutor(),
        tracer=Tracer(stream=not args.json),
        budget=Budget(),
    )
    result = agent.run(args.objective, required_categories=required)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print("\n=== RESULT ===")
        print(f"success: {result.success}")
        print(f"answer : {result.answer}")
        print(f"trace  : {result.trace}")
        print(f"budget : {result.budget}")
        if result.pending_review:
            print(f"pending_review: {len(result.pending_review)} action(s) need approval")
    return 0 if result.success else 2


if __name__ == "__main__":
    sys.exit(main())
