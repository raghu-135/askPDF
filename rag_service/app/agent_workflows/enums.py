"""String enums for agent workflow contracts and runtime states."""

from enum import Enum


class InterruptStatus(str, Enum):
    PENDING = "pending"
    RESUMED = "resumed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AgentRunResumeAction(str, Enum):
    APPROVE = "approve"
    APPROVE_SELECTED = "approve_selected"
    REJECT = "reject"
    EDIT = "edit"
    CONTINUE_WITHOUT = "continue_without"


class HitlMode(str, Enum):
    APPROVAL = "approval"
    CHOICE = "choice"
    REVIEW = "review"


class HitlPhase(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    INSIDE_TOOL = "inside_tool"


class HitlSelectionMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"
    SINGLE_OR_MULTI = "single_or_multi"


class RouterRoute(str, Enum):
    DOCUMENT = "document"
    MEMORY = "memory"
    TIMELINE = "timeline"
    WEB = "web"
    DIRECT = "direct"
    CLARIFY = "clarify"


class PlannerRoute(str, Enum):
    EXECUTE = "execute"
    DIRECT = "direct"
    CLARIFY = "clarify"


class EvaluatorRoute(str, Enum):
    ANSWER = "answer"
    REPLAN = "replan"
    ANSWER_BUDGET_EXHAUSTED = "answer_budget_exhausted"


class RouteFunctionId(str, Enum):
    ROUTER = "router_route"
    PLANNER = "planner_route"
    EVALUATOR = "evaluator_route"
    HITL_GATE = "hitl_gate_route"


class WorkflowNodeType(str, Enum):
    CONTEXT_LOADER = "context_loader"
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVAL_WORKER = "retrieval_worker"
    MEMORY_WORKER = "memory_worker"
    TIMELINE_WORKER = "timeline_worker"
    WEB_WORKER = "web_worker"
    EVIDENCE_EVALUATOR = "evidence_evaluator"
    REPLANNER = "replanner"
    DIRECT_ANSWER = "direct_answer"
    SYNTHESIZER = "synthesizer"
    FINALIZER = "finalizer"
    HITL_GATE = "hitl_gate"


class ToolName(str, Enum):
    GET_THREAD_SHAPE = "get_thread_shape"
    SEARCH_DOCUMENTS = "search_documents"
    SEARCH_DOCUMENT_BY_ID = "search_document_by_id"
    SEARCH_CONVERSATION_HISTORY = "search_conversation_history"
    SEARCH_THREAD_TIMELINE = "search_thread_timeline"
    SEARCH_WEB = "search_web"
    WIKIPEDIA = "wikipedia"
    WIKIDATA = "wikidata"
    ARXIV = "arxiv"
    PUB_MED = "pub_med"
    PUBMED = "pubmed"
    SEMANTIC_SCHOLAR_LEGACY = "semanticscholar"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    STACK_EXCHANGE = "stack_exchange"
    YAHOO_FINANCE_NEWS = "yahoo_finance_news"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"


class GraphSentinel(str, Enum):
    START = "START"
    END = "END"


class NodeCategory(str, Enum):
    CONTEXT = "context"
    CONTROL = "control"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    TIMELINE = "timeline"
    WEB = "web"
    ANSWER = "answer"
    HUMAN_REVIEW = "human_review"
    EXTERNAL_RESEARCH = "external_research"


class EvidenceKind(str, Enum):
    DOCUMENT = "document"
    MEMORY = "memory"
    TIMELINE = "timeline"
    WEB = "web"


class NodeCapability(str, Enum):
    CONTEXT_PREFETCH = "context.prefetch"
    ROUTE_INTENT = "route.intent"
    CLARIFY = "clarify"
    PLAN_EXECUTION = "plan.execution"
    PLAN_REPLAN = "plan.replan"
    RETRIEVAL_DOCUMENT = "retrieval.document"
    RETRIEVAL_MEMORY = "retrieval.memory"
    RETRIEVAL_TIMELINE = "retrieval.timeline"
    RETRIEVAL_WEB = "retrieval.web"
    EXTERNAL_RESEARCH = "external_research"
    EVALUATE_EVIDENCE = "evaluate.evidence"
    ANSWER_DIRECT = "answer.direct"
    ANSWER_SYNTHESIZE = "answer.synthesize"
    ANSWER_FINAL = "answer.final"
    HITL_INTERRUPT = "hitl.interrupt"


class ContextPolicyMode(str, Enum):
    PREFETCH = "prefetch"
    ROUTE = "route"
    PLAN = "plan"
    APPEND_EVIDENCE = "append_evidence"
    EVALUATE_EVIDENCE = "evaluate_evidence"
    ASSEMBLE_ANSWER = "assemble_answer"
    FINALIZE = "finalize"
    INTERRUPT = "interrupt"


class ContextBudget(str, Enum):
    REQUEST = "request"
    BOUNDED = "bounded"
    BOUNDED_PREFETCH = "bounded_prefetch"
    TOOL_QUERY = "tool_query"
    EVIDENCE_PACKET = "evidence_packet"
    BOUNDED_EVIDENCE = "bounded_evidence"
    BOUNDED_SUMMARY = "bounded_summary"
    DECISION = "decision"
    ANSWER = "answer"


class TraceSpanKind(str, Enum):
    CONTEXT = "context"
    CONTROL = "control"
    TOOL_WORKER = "tool_worker"
    ANSWER = "answer"
    HUMAN_REVIEW = "human_review"


class TracePayloadMode(str, Enum):
    BOUNDED = "bounded"


class NodeEventStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class TraceStatus(str, Enum):
    ERROR = "error"


class DebugGraphNodeStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    INACTIVE = "inactive"
    ERROR = "error"


class PromptProfile(str, Enum):
    ROUTER = "router"
    PLANNER = "planner"
    EVALUATOR_REPLANNER = "evaluator_replanner"


class AgentCheckpointerMode(str, Enum):
    MEMORY = "memory"
    POSTGRES = "postgres"


class EvidenceCompressionMode(str, Enum):
    NONE = "none"
    COMPACT = "compact"


class ToolContractId(str, Enum):
    THREAD_SHAPE = "thread_shape"
    DOCUMENT_EVIDENCE = "document_evidence"
    FOCUSED_DOCUMENT_EVIDENCE = "focused_document_evidence"
    DEEP_MEMORY = "deep_memory"
    THREAD_TIMELINE = "thread_timeline"
    LIVE_WEB_RECON = "live_web_recon"
    WIKIPEDIA_REFERENCE = "wikipedia_reference"
    WIKIDATA_REFERENCE = "wikidata_reference"
    ARXIV_RESEARCH = "arxiv_research"
    PUBMED_RESEARCH = "pubmed_research"
    SEMANTIC_SCHOLAR_RESEARCH = "semantic_scholar_research"
    STACKEXCHANGE_REFERENCE = "stackexchange_reference"
    YAHOO_FINANCE_NEWS = "yahoo_finance_news"
    CLARIFY_INTENT = "clarify_intent"


ROUTER_ROUTES = {route.value for route in RouterRoute}
PLANNER_ROUTES = {route.value for route in PlannerRoute}
EVALUATOR_ROUTES = {route.value for route in EvaluatorRoute}
HITL_ACTIONS = {action.value for action in AgentRunResumeAction}
RESUME_ACTIONS = {
    AgentRunResumeAction.APPROVE.value,
    AgentRunResumeAction.APPROVE_SELECTED.value,
    AgentRunResumeAction.EDIT.value,
    AgentRunResumeAction.CONTINUE_WITHOUT.value,
}
HITL_PHASES = {phase.value for phase in HitlPhase}
HITL_MODES = {mode.value for mode in HitlMode}
HITL_SELECTION_MODES = {mode.value for mode in HitlSelectionMode}
TERMINAL_INTERRUPT_STATUSES = {
    InterruptStatus.RESUMED.value,
    InterruptStatus.REJECTED.value,
    InterruptStatus.EXPIRED.value,
}
