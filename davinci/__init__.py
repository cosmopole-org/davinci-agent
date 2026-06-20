"""Davinci — an orchestration-first, enterprise-grade agent runtime.

Public surface:

    from davinci import DavinciAgent, ToolRegistry, PermissionEngine, Budget

The package is intentionally stdlib-only so it runs unmodified inside a slim
Docker creature deployed on a Caspar node.
"""

from .attachments import (
    Attachment,
    attachment_from_file,
    materialize_attachments,
    INLINE_MAX_BYTES,
)
from .engine import (
    ActionProposal,
    DavinciAgent,
    EchoExecutor,
    HeuristicReasoner,
    Reflection,
    RunResult,
    ToolExecutor,
    ToolResult,
)
from .llm import (
    AnthropicClient,
    GeminiClient,
    GrokClient,
    LLMClient,
    OpenAIClient,
    OpenAICompatibleClient,
    client_from_config,
    make_client,
)
from .mcp import ToolDescriptor, ToolRegistry
from .memory import EpisodicMemory, InstructionMemory, WorkingMemory, compact_messages
from .observability import Budget, Tracer, mask_secrets
from .reasoner import LLMReasoner, reasoner_from_config
from .permissions import Decision, Outcome, PermissionEngine, PermissionMode, Risk, ToolAction
from .planning import Plan, Planner, PlanStep, StepStatus

__version__ = "0.2.0"

__all__ = [
    "DavinciAgent", "RunResult", "ToolExecutor", "ToolResult", "ActionProposal",
    "HeuristicReasoner", "EchoExecutor", "Reflection",
    "ToolRegistry", "ToolDescriptor",
    "WorkingMemory", "EpisodicMemory", "InstructionMemory", "compact_messages",
    "Tracer", "Budget", "mask_secrets",
    "PermissionEngine", "PermissionMode", "Risk", "Outcome", "Decision", "ToolAction",
    "Plan", "Planner", "PlanStep", "StepStatus",
    "Attachment", "attachment_from_file", "materialize_attachments", "INLINE_MAX_BYTES",
    "LLMReasoner", "reasoner_from_config",
    "LLMClient", "GeminiClient", "AnthropicClient", "OpenAIClient",
    "OpenAICompatibleClient", "GrokClient", "make_client", "client_from_config",
    "__version__",
]
