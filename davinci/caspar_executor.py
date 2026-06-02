"""Caspar-backed tool executor for the Davinci engine.

Bridges the engine's pluggable ``ToolExecutor`` to real Caspar tool VMs: when
the engine decides to use a tool, this executor emits a ``signal`` host call to
the corresponding pre-running tool container and returns the dispatch record.

This keeps the brain (planning/permissions/budget) identical whether running
offline or against a live Caspar cluster — only the executor changes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .engine import ToolResult
from .mcp import ToolDescriptor
from .permissions import ToolAction


class CasparSignalExecutor:
    """Execute tools by signalling Caspar tool VMs via the VMM host API."""

    def __init__(self, caspar_registry: Any, client: Optional[Any] = None,
                 user_id: str = "davinci-agent") -> None:
        # Lazy import keeps caspar_orchestrator optional for pure-offline use.
        from caspar_orchestrator import CasparVmmClient

        self.caspar_registry = caspar_registry
        self.client = client or CasparVmmClient()
        self.user_id = user_id
        # Index Caspar capabilities by their MCP-namespaced name.
        self._by_name: Dict[str, Any] = {}
        for cap in caspar_registry.list_tools():
            self._by_name[f"caspar__{cap.tool_id}"] = cap
            self._by_name[cap.tool_id] = cap

    def execute(self, tool: ToolDescriptor, action: ToolAction) -> ToolResult:
        cap = self._by_name.get(tool.name)
        if cap is None:
            return ToolResult(ok=False, output=None, error=f"unknown Caspar tool {tool.name}")
        category = cap.categories[0] if cap.categories else action.category
        function_name = cap.functions.get(category, "invoke")
        task = str(action.args.get("task", ""))
        try:
            dispatch = self.client.signal_tool(
                tool=cap, category=category, function_name=function_name,
                task=task, user_id=self.user_id,
            )
        except Exception as exc:  # network/transport failures are recoverable
            return ToolResult(ok=False, output=None, error=str(exc))
        return ToolResult(ok=True, output={
            "dispatched": True,
            "tool": tool.name,
            "category": category,
            "function": function_name,
            "response_point": cap.response_point,
            "signal": dispatch,
        })
