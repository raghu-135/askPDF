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


class HitlInterruptType(str, Enum):
    HUMAN_REVIEW = "human_review"
    OPTION_REVIEW = "option_review"


class HitlRejectBehavior(str, Enum):
    RESUME = "resume"


class RouterRoute(str, Enum):
    DOCUMENT = "document"
    THREAD_CONVERSATION_HISTORY = "thread_conversation_history"
    DURABLE_MEMORY = "durable_memory"
    THREAD_EVENTS = "thread_events"
    WEB = "web"
    COMPOUND = "compound"
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


class AnswerQualityRoute(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    FINALIZE_CAUTIOUS = "finalize_cautious"


class RouteFunctionId(str, Enum):
    ROUTER = "router_route"
    PLANNER = "planner_route"
    EVALUATOR = "evaluator_route"
    HITL_GATE = "hitl_gate_route"
    PARALLEL_DISPATCH = "parallel_dispatch_route"
    SERIAL_DISPATCH = "serial_dispatch_route"
    ANSWER_QUALITY = "answer_quality_route"


class WorkflowNodeType(str, Enum):
    CONTEXT_LOADER = "context_loader"
    ROUTER = "router"
    PLANNER = "planner"
    RETRIEVAL_WORKER = "retrieval_worker"
    THREAD_CONVERSATION_HISTORY_WORKER = "thread_conversation_history_worker"
    DURABLE_MEMORY_WORKER = "durable_memory_worker"
    THREAD_EVENTS_WORKER = "thread_events_worker"
    WEB_WORKER = "web_worker"
    EVIDENCE_EVALUATOR = "evidence_evaluator"
    REPLANNER = "replanner"
    DIRECT_ANSWER = "direct_answer"
    SYNTHESIZER = "synthesizer"
    FINALIZER = "finalizer"
    HITL_GATE = "hitl_gate"
    PARALLEL_DISPATCH = "parallel_dispatch"
    SERIAL_DISPATCH = "serial_dispatch"
    AGGREGATOR = "aggregator"
    ANSWER_EVALUATOR = "answer_evaluator"
    ANSWER_REVISER = "answer_reviser"


class ToolName(str, Enum):
    GET_THREAD_SHAPE = "get_thread_shape"
    SEARCH_DOCUMENTS = "search_documents"
    SEARCH_DOCUMENT_BY_ID = "search_document_by_id"
    SEARCH_THREAD_CONVERSATION_HISTORY = "search_thread_conversation_history"
    SEARCH_DURABLE_MEMORY = "search_durable_memory"
    SEARCH_THREAD_EVENTS = "search_thread_events"
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
    THREAD_CONVERSATION_HISTORY = "thread_conversation_history"
    DURABLE_MEMORY = "durable_memory"
    THREAD_EVENTS = "thread_events"
    WEB = "web"
    ANSWER = "answer"
    HUMAN_REVIEW = "human_review"
    EXTERNAL_RESEARCH = "external_research"


class EvidenceKind(str, Enum):
    DOCUMENT = "document"
    THREAD_CONVERSATION_HISTORY = "thread_conversation_history"
    DURABLE_MEMORY = "durable_memory"
    THREAD_EVENTS = "thread_events"
    WEB = "web"


class NodeCapability(str, Enum):
    CONTEXT_PREFETCH = "context.prefetch"
    ROUTE_INTENT = "route.intent"
    CLARIFY = "clarify"
    PLAN_EXECUTION = "plan.execution"
    PLAN_REPLAN = "plan.replan"
    RETRIEVAL_DOCUMENT = "retrieval.document"
    RETRIEVAL_THREAD_CONVERSATION_HISTORY = "retrieval.thread_conversation_history"
    RETRIEVAL_DURABLE_MEMORY = "retrieval.durable_memory"
    RETRIEVAL_THREAD_EVENTS = "retrieval.thread_events"
    RETRIEVAL_WEB = "retrieval.web"
    EXTERNAL_RESEARCH = "external_research"
    EVALUATE_EVIDENCE = "evaluate.evidence"
    ANSWER_DIRECT = "answer.direct"
    ANSWER_SYNTHESIZE = "answer.synthesize"
    ANSWER_FINAL = "answer.final"
    HITL_INTERRUPT = "hitl.interrupt"
    PARALLEL_DISPATCH = "parallel.dispatch"
    PARALLEL_AGGREGATE = "parallel.aggregate"
    SERIAL_DISPATCH = "serial.dispatch"
    EVALUATE_ANSWER = "evaluate.answer"
    REVISE_ANSWER = "answer.revise"


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


class WorkflowRuntimeKind(str, Enum):
    COMPILED_RAG = "compiled_rag"


class EvidenceCompressionMode(str, Enum):
    NONE = "none"
    COMPACT = "compact"


class PlannerRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolContractId(str, Enum):
    THREAD_SHAPE = "thread_shape"
    DOCUMENT_EVIDENCE = "document_evidence"
    FOCUSED_DOCUMENT_EVIDENCE = "focused_document_evidence"
    THREAD_CONVERSATION_HISTORY = "thread_conversation_history"
    DURABLE_MEMORY = "durable_memory"
    THREAD_EVENTS = "thread_events"
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
ANSWER_QUALITY_ROUTES = {route.value for route in AnswerQualityRoute}
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
