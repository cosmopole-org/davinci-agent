"""Runtime feature model for Davinci as a generalist agent brain.

This module now combines:
1) Environment-backed checks (Docker VM + external connectivity), and
2) Feature-parity modeling for modern agentic systems (planning, memory,
   tool use, HITL, multi-agent/VM orchestration).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import shutil
import socket
import subprocess
from typing import Dict, List, Optional

from caspar_orchestrator import bootstrap_default_registry, summarize_tool_coverage


@dataclass(frozen=True)
class FeatureStatus:
    """One capability with implementation and runtime support details."""

    key: str
    supported: bool
    scope: str
    reason: str
    implementation: str
    inspired_by: List[str]
    evidence: Optional[str] = None


class AgenticRuntime:
    """Detect and describe broad agentic capabilities for Davinci."""

    def __init__(self, workspace: str = "/workspace") -> None:
        self.workspace = workspace
        self.registry = bootstrap_default_registry()

    def detect(self) -> List[FeatureStatus]:
        checks = [
            self._planning_reasoning,
            self._tool_use_and_function_calls,
            self._multi_vm_orchestration,
            self._human_in_the_loop,
            self._memory_state,
            self._retrieval_augmented_generation,
            self._code_and_shell_execution,
            self._filesystem_state,
            self._version_control_and_delivery,
            self._web_research_and_live_data,
            self._ui_and_browser_automation,
            self._integrations_and_mini_apps,
            self._background_long_running_work,
            self._safety_guardrails,
            self._structured_outputs,
            self._time_awareness,
        ]
        return [c() for c in checks]

    def _planning_reasoning(self) -> FeatureStatus:
        return FeatureStatus(
            key="planning_reasoning",
            supported=True,
            scope="brain_vm",
            reason="Davinci acts as the central planner, decomposing objectives into ordered actions.",
            implementation="Task decomposition + execution plan generation.",
            inspired_by=["Claude Code", "OpenAI Agents", "LangGraph"],
            evidence="Execution plans generated before tool dispatch.",
        )

    def _tool_use_and_function_calls(self) -> FeatureStatus:
        return FeatureStatus(
            key="tool_use_function_calling",
            supported=True,
            scope="brain_vm",
            reason="Brain routes tasks to explicit tool interfaces instead of free-form guessing.",
            implementation="Structured capability selection via categories and risk levels.",
            inspired_by=["Claude Code", "OpenAI function calling"],
            evidence="Caspar host-function mappings + DavinciPlanner abstractions.",
        )

    def _multi_vm_orchestration(self) -> FeatureStatus:
        vm_count = len(self.registry.list_vms())
        supported = vm_count >= 3
        return FeatureStatus(
            key="multi_vm_orchestration",
            supported=supported,
            scope="caspar_vm_to_vm",
            reason="Brain can orchestrate specialized Docker VMs through Caspar VMM host-function APIs."
            if supported
            else "Not enough worker VMs registered for broad orchestration.",
            implementation="Caspar VMM host-call model + per-VM tool catalogs + routing planner.",
            inspired_by=["Multi-agent architectures", "MCP-style tool registries"],
            evidence=f"registered_vms={vm_count}",
        )

    def _human_in_the_loop(self) -> FeatureStatus:
        return FeatureStatus(
            key="human_in_the_loop",
            supported=True,
            scope="workflow",
            reason="Risky or incomplete plans can be flagged for user review before execution.",
            implementation="Plan metadata includes human_review_required signal.",
            inspired_by=["LangGraph HITL", "Claude approval workflows"],
            evidence="Low-tolerance or high-risk steps require review.",
        )

    def _memory_state(self) -> FeatureStatus:
        return FeatureStatus(
            key="memory_state_management",
            supported=True,
            scope="brain_vm",
            reason="Agent persists context in artifacts and git history.",
            implementation="Workspace files + commit timeline as durable memory layers.",
            inspired_by=["Long-context coding agents"],
            evidence=self.workspace,
        )

    def _retrieval_augmented_generation(self) -> FeatureStatus:
        categories = summarize_tool_coverage(self.registry)
        supported = "knowledge_retrieval" in categories
        return FeatureStatus(
            key="retrieval_augmented_generation",
            supported=supported,
            scope="worker_vm",
            reason="Dedicated retrieval capability exists for external/document knowledge grounding."
            if supported
            else "No retrieval tool registered.",
            implementation="Route retrieval requests to vector_search tool on data VM.",
            inspired_by=["RAG agents", "Assistant retrieval pipelines"],
            evidence=", ".join(categories.get("knowledge_retrieval", [])) or None,
        )

    def _code_and_shell_execution(self) -> FeatureStatus:
        shell_path = shutil.which("bash")
        return FeatureStatus(
            key="code_shell_execution",
            supported=bool(shell_path),
            scope="docker_vm",
            reason="Local shell execution is available for coding, tests, and automation."
            if shell_path
            else "No shell binary found.",
            implementation="Run commands and scripts inside Docker VM.",
            inspired_by=["Claude Code", "Code-interpreter style agents"],
            evidence=shell_path,
        )

    def _filesystem_state(self) -> FeatureStatus:
        writable = os.path.exists(self.workspace) and os.access(self.workspace, os.W_OK)
        return FeatureStatus(
            key="filesystem_rw",
            supported=writable,
            scope="docker_vm",
            reason="Agent can update repo files, configs, and artifacts."
            if writable
            else "Workspace is not writable.",
            implementation="Direct file IO under workspace root.",
            inspired_by=["Coding agents", "Autonomous build/test loops"],
            evidence=self.workspace,
        )

    def _version_control_and_delivery(self) -> FeatureStatus:
        git = shutil.which("git")
        if not git:
            return FeatureStatus(
                key="version_control_delivery",
                supported=False,
                scope="docker_vm",
                reason="git executable missing.",
                implementation="N/A",
                inspired_by=["DevOps agents"],
            )

        proc = subprocess.run(
            [git, "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0 and proc.stdout.strip() == "true"
        return FeatureStatus(
            key="version_control_delivery",
            supported=ok,
            scope="docker_vm",
            reason="Repository context available for commits/PR workflow."
            if ok
            else "Not currently inside a git work tree.",
            implementation="git-vm unified tool routed via signalPoint request/response channels.",
            inspired_by=["Claude Code", "CI-aware coding agents"],
            evidence=proc.stdout.strip() or proc.stderr.strip(),
        )

    def _web_research_and_live_data(self) -> FeatureStatus:
        try:
            socket.gethostbyname("example.com")
            dns_ok = True
        except OSError:
            dns_ok = False

        curl = shutil.which("curl")
        http_ok = False
        if curl:
            proc = subprocess.run(
                [curl, "-I", "-sS", "https://example.com"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            http_ok = proc.returncode == 0 and "HTTP/" in proc.stdout

        supported = dns_ok and http_ok
        return FeatureStatus(
            key="web_research_live_data",
            supported=supported,
            scope="outside_vm",
            reason="External web interaction is available for current-state grounding."
            if supported
            else "External connectivity partially unavailable.",
            implementation="web-vm request/response signaling via Caspar host functions + direct HTTP checks.",
            inspired_by=["Research agents", "Live browsing copilots"],
            evidence=f"dns_ok={dns_ok}, http_ok={http_ok}",
        )

    def _ui_and_browser_automation(self) -> FeatureStatus:
        categories = summarize_tool_coverage(self.registry)
        supported = "ui_automation" in categories
        return FeatureStatus(
            key="ui_browser_automation",
            supported=supported,
            scope="worker_vm",
            reason="Browser/UI automation can be offloaded to a dedicated worker VM."
            if supported
            else "No browser automation tool registered.",
            implementation="browser_automation routed through signalPoint request/response flow.",
            inspired_by=["Computer-use agents", "RPA workflows"],
            evidence=", ".join(categories.get("ui_automation", [])) or None,
        )

    def _integrations_and_mini_apps(self) -> FeatureStatus:
        categories = summarize_tool_coverage(self.registry)
        supported = "integrations" in categories
        return FeatureStatus(
            key="saas_integrations_mini_apps",
            supported=supported,
            scope="caspar_miniapps_vm",
            reason="Third-party product actions can run through mini-app connectors."
            if supported
            else "No integration connectors registered.",
            implementation="mini-app connectors routed via signalPoint request/response flow.",
            inspired_by=["Action agents", "Enterprise copilots"],
            evidence=", ".join(categories.get("integrations", [])) or None,
        )

    def _background_long_running_work(self) -> FeatureStatus:
        return FeatureStatus(
            key="background_long_running_tasks",
            supported=True,
            scope="docker_vm",
            reason="Long processes can be run and polled incrementally.",
            implementation="TTY session + progressive polling pattern.",
            inspired_by=["Interactive coding loops"],
            evidence="Supports asynchronous command follow-up.",
        )

    def _safety_guardrails(self) -> FeatureStatus:
        return FeatureStatus(
            key="safety_guardrails",
            supported=True,
            scope="workflow",
            reason="Risk-tagged tools and review gates enforce controlled execution.",
            implementation="Risk levels + human review trigger in plan policy.",
            inspired_by=["Guardrailed agent frameworks"],
            evidence="high-risk tools set review_required in execution plans.",
        )

    def _structured_outputs(self) -> FeatureStatus:
        return FeatureStatus(
            key="structured_outputs",
            supported=True,
            scope="brain_vm",
            reason="Capabilities and plans are JSON serializable for interop.",
            implementation="Dataclass-backed reports.",
            inspired_by=["Schema-first agent APIs"],
            evidence="asdict + JSON serialization path.",
        )

    def _time_awareness(self) -> FeatureStatus:
        return FeatureStatus(
            key="time_awareness",
            supported=True,
            scope="brain_vm",
            reason="Agent can stamp all decisions with absolute UTC time.",
            implementation="UTC timestamps in reports and metadata.",
            inspired_by=["Audit-friendly orchestration systems"],
            evidence=datetime.now(timezone.utc).isoformat(),
        )


def capability_report() -> Dict[str, object]:
    runtime = AgenticRuntime()
    features = runtime.detect()
    supported = [f.key for f in features if f.supported]
    unsupported = [f.key for f in features if not f.supported]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": runtime.workspace,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "supported_features": supported,
        "unsupported_features": unsupported,
        "features": [asdict(f) for f in features],
        "vm_tool_coverage": summarize_tool_coverage(runtime.registry),
    }


if __name__ == "__main__":
    print(json.dumps(capability_report(), indent=2))
