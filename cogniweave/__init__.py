"""cogniweave — multi-agent orchestration on a cognitive architecture."""

from .cognition.memory import (
    EpisodicMemory,
    MemorySystem,
    ProceduralMemory,
    SemanticMemory,
    WorkingMemory,
)
from .core.agent import AgentOutput, CognitiveAgent
from .core.blackboard import Blackboard
from .core.tools import Tool, ToolRegistry, tool
from .core.tracing import BudgetExceeded, Tracer
from .crew import Crew, build_debate_panel, build_research_crew
from .llm import MockLLM, get_llm
from .orchestration.blackboard_pattern import BlackboardOrchestrator
from .orchestration.debate import DebateOrchestrator
from .orchestration.hierarchical import HierarchicalOrchestrator
from .orchestration.plan import Plan, PlanTask, Review
from .orchestration.result import RunResult
from .orchestration.sequential import SequentialOrchestrator
from .runner import PATTERNS, run_pattern

__version__ = "0.1.0"

__all__ = [
    "PATTERNS",
    "AgentOutput",
    "Blackboard",
    "BlackboardOrchestrator",
    "BudgetExceeded",
    "CognitiveAgent",
    "Crew",
    "DebateOrchestrator",
    "EpisodicMemory",
    "HierarchicalOrchestrator",
    "MemorySystem",
    "MockLLM",
    "Plan",
    "PlanTask",
    "ProceduralMemory",
    "Review",
    "RunResult",
    "SemanticMemory",
    "SequentialOrchestrator",
    "Tool",
    "ToolRegistry",
    "Tracer",
    "WorkingMemory",
    "build_debate_panel",
    "build_research_crew",
    "get_llm",
    "run_pattern",
    "tool",
]
