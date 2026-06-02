"""MCP-style tool registry with deferred schemas and tool search.

Modern agents standardise tooling on the Model Context Protocol: tools are
described by JSON-Schema, discovered from registries, namespaced, and — crucially
for context economy — their full schemas are *deferred* (only names are known
until a tool is actually needed). This module provides:

* ``ToolDescriptor`` — an MCP-shaped tool (name, description, input schema,
  risk, category) with a lazy schema loader.
* ``ToolRegistry``   — namespaced registration + keyword/required-term search
  that returns ranked matches, à la Claude Code's ToolSearch.

It interoperates with ``caspar_orchestrator.discover_tool_capabilities`` so the
same Caspar point catalog that drives signalling also feeds the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ToolDescriptor:
    name: str                       # namespaced, e.g. "caspar__git_tool"
    description: str
    category: str = "general"
    risk: str = "medium"
    requires_network: bool = False
    server: str = "local"
    # Deferred: the full input schema is only materialised on demand so a large
    # catalog doesn't consume context until a tool is selected.
    _schema_loader: Optional[Callable[[], Dict[str, Any]]] = field(default=None, repr=False)
    _schema: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def schema(self) -> Dict[str, Any]:
        if self._schema is None:
            self._schema = self._schema_loader() if self._schema_loader else {"type": "object", "properties": {}}
        return self._schema

    def stub(self) -> Dict[str, Any]:
        """The cheap, always-loaded view (no schema)."""
        return {"name": self.name, "description": self.description,
                "category": self.category, "risk": self.risk, "server": self.server}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDescriptor] = {}

    def register(self, tool: ToolDescriptor) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDescriptor]:
        return self._tools.get(name)

    def all(self) -> List[ToolDescriptor]:
        return list(self._tools.values())

    def by_category(self, category: str) -> List[ToolDescriptor]:
        return [t for t in self._tools.values() if t.category == category]

    def categories(self) -> List[str]:
        return sorted({t.category for t in self._tools.values()})

    def search(self, query: str, *, max_results: int = 5) -> List[ToolDescriptor]:
        """Rank tools by query relevance.

        Supports the "+term" required-term syntax and "select:a,b" exact lookup,
        mirroring the deferred-tool search convention.
        """
        query = query.strip()
        if query.startswith("select:"):
            names = [n.strip() for n in query[len("select:"):].split(",") if n.strip()]
            return [self._tools[n] for n in names if n in self._tools]

        required: List[str] = []
        optional: List[str] = []
        for term in query.lower().split():
            (required if term.startswith("+") else optional).append(term.lstrip("+"))

        scored: List[tuple] = []
        for tool in self._tools.values():
            haystack = f"{tool.name} {tool.description} {tool.category}".lower()
            if any(req not in haystack for req in required):
                continue
            score = sum(haystack.count(term) for term in optional) + len(required)
            if score > 0 or not optional:
                scored.append((score, tool))
        scored.sort(key=lambda x: (-x[0], x[1].name))
        return [t for _, t in scored[:max_results]]

    # -- interop with caspar_orchestrator -----------------------------------
    @classmethod
    def from_caspar_registry(cls, caspar_registry: Any, server: str = "caspar") -> "ToolRegistry":
        """Build an MCP registry from a DavinciToolRegistry (Caspar tools)."""
        reg = cls()
        for cap in caspar_registry.list_tools():
            category = cap.categories[0] if cap.categories else "general"
            schemas = getattr(cap, "function_schemas", {}) or {}

            def loader(_schemas=schemas, _cat=category) -> Dict[str, Any]:
                spec = _schemas.get(_cat) or next(iter(_schemas.values()), None)
                if spec and isinstance(spec, dict):
                    return spec.get("input_schema") or {"type": "object", "properties": {}}
                return {"type": "object", "properties": {}}

            reg.register(ToolDescriptor(
                name=f"{server}__{cap.tool_id}",
                description=cap.description,
                category=category,
                risk=cap.risk_level,
                requires_network=cap.requires_network,
                server=server,
                _schema_loader=loader,
            ))
        return reg
