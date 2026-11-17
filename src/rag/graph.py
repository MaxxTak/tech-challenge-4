"""LangGraph StateGraph orchestration with early exits, XML fencing, and citation repair."""

from __future__ import annotations

import re
from html import escape
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.config import Settings, settings
from src.rag.expander import expand_query, normalizar_variacoes
from src.rag.prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    SYNTHESIS_USER_PROMPT,
    VERBATIM_INSUFFICIENT_REFUSAL,
)
from src.rag.retriever import HybridRetriever
from src.rag.router import route_query
from src.rag.state import QueryIntent, RAGState, RetrievedEvidence
from src.security.guardrails import detectar_injection, mascarar
from src.storage.parquet_store import ParquetAnalyticsStore

_CITATION_RE = re.compile(r"\[([a-f0-9]{16,36}|[A-Za-z0-9_-]{8,40})\]")


class _UnconfiguredLLM(Runnable):
    """Deferred placeholder allowing FastAPI/Streamlit startup without API keys."""

    def with_structured_output(self, schema: Any) -> Any:
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Nenhuma chave LLM configurada. Defina GOOGLE_API_KEY ou OPENAI_API_KEY no .env."
        )


def build_resilient_llm(temperature: float = 0.0, cfg: Settings = settings) -> Any:
    """Create primary Gemini LLM with automatic OpenAI fallback."""
    if cfg.GOOGLE_API_KEY:
        primary = ChatGoogleGenerativeAI(
            model=cfg.GOOGLE_CHAT_MODEL,
            temperature=temperature,
            google_api_key=cfg.GOOGLE_API_KEY,
            request_timeout=20,
            retries=1,
        )
        if cfg.OPENAI_API_KEY:
            fallback = ChatOpenAI(
                model=cfg.OPENAI_CHAT_MODEL,
                temperature=temperature,
                timeout=20,
                api_key=cfg.OPENAI_API_KEY,
            )
            return primary.with_fallbacks([fallback])
        return primary

    if cfg.OPENAI_API_KEY:
        return ChatOpenAI(
            model=cfg.OPENAI_CHAT_MODEL,
            temperature=temperature,
            timeout=20,
            api_key=cfg.OPENAI_API_KEY,
        )

    return _UnconfiguredLLM()


def _format_chat_history(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "(primeira interação da sessão)"
    lines = []
    for turn in chat_history[-6:]:
        role = "Analista" if turn.get("role") == "human" else "Assistente"
        lines.append(f"- {role}: {turn.get('content', '')!s}")
    return "\n".join(lines)


def _format_evidence_xml(evidences: list[RetrievedEvidence]) -> str:
    blocks = []
    for ev in evidences:
        late_str = "true" if ev.is_late_delivery else "false"
        body = escape(ev.raw_comment_masked or ev.text)
        blocks.append(
            f'<evidence rank="{ev.rank}" review_id="{escape(ev.review_id)}" '
            f'score="{ev.review_score}" category="{escape(ev.product_category)}" '
            f'order_status="{escape(ev.order_status)}" '
            f'late="{late_str}" delay_days="{ev.delivery_delay_days}" '
            f'month="{escape(ev.purchase_year_month)}">\n{body}\n</evidence>'
        )
    return "\n\n".join(blocks)


def _describe_active_filters(intent_dict: dict[str, Any] | None) -> str:
    if not intent_dict:
        return ""
    parts: list[str] = []
    for key in (
        "score_min",
        "score_max",
        "product_category",
        "order_status",
        "is_late_delivery",
        "date_start",
        "date_end",
    ):
        val = intent_dict.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def build_rag_graph(
    *,
    retriever: HybridRetriever,
    parquet_store: ParquetAnalyticsStore,
    llm: Any | None = None,
    cfg: Settings = settings,
) -> Any:
    """Compile the 8-node LangGraph StateGraph for Olist VoC RAG."""
    active_llm = llm if llm is not None else build_resilient_llm(temperature=0.0, cfg=cfg)

    def guardrail_entrada(state: RAGState) -> dict[str, Any]:
        question = state["question"]
        inj = detectar_injection(question)
        sanitized, hashes = mascarar(question)
        if inj["blocked"]:
            reason = ", ".join(inj["motivos"])
            return {
                "sanitized_question": sanitized,
                "pii_hashes": hashes,
                "guardrail_blocked": True,
                "block_reason": reason,
                "status": "blocked",
                "trilha": [f"guardrail_entrada:blocked ({reason})"],
            }
        return {
            "sanitized_question": sanitized,
            "pii_hashes": hashes,
            "guardrail_blocked": False,
            "block_reason": None,
            "trilha": ["guardrail_entrada:ok"],
        }

    def rotear_consulta(state: RAGState) -> dict[str, Any]:
        intent = route_query(
            state["sanitized_question"],
            state.get("chat_history", []),
            active_llm,
            parquet_store.valid_categories,
            parquet_store.valid_order_statuses,
            state.get("explicit_filters"),
        )
        return {
            "intent": intent.model_dump(),
            "trilha": [f"rotear_consulta:{intent.intent_type}"],
        }

    def expandir_consulta(state: RAGState) -> dict[str, Any]:
        intent = QueryIntent(**state["intent"])
        if intent.query_variations:
            variations = normalizar_variacoes(
                intent.semantic_query,
                intent.query_variations,
                max_variations=2,
            )
            return {
                "query_variations": variations,
                "trilha": [f"expandir_consulta:reused_from_router ({len(variations)} queries)"],
            }
        variations = expand_query(intent.semantic_query, active_llm, qtd=2)
        return {
            "query_variations": variations,
            "trilha": [f"expandir_consulta:{len(variations)} queries"],
        }

    def recuperar_hibrido(state: RAGState) -> dict[str, Any]:
        intent = QueryIntent(**state["intent"])
        evidences, max_dense, agg_stats = retriever.retrieve(
            intent,
            state["query_variations"],
            top_k=state.get("top_k", cfg.DEFAULT_TOP_K),
        )
        return {
            "evidences": evidences,
            "max_dense_score": max_dense,
            "aggregate_stats": agg_stats,
            "trilha": [f"recuperar_hibrido:docs={len(evidences)},max_dense={max_dense:.3f}"],
        }

    def avaliar_evidencias(state: RAGState) -> dict[str, Any]:
        evidences = state.get("evidences", [])
        max_dense = float(state.get("max_dense_score", 0.0))
        ok = len(evidences) > 0 and max_dense >= cfg.SIMILARITY_THRESHOLD
        return {
            "confidence_ok": ok,
            "trilha": [
                (
                    f"avaliar_evidencias:{'ok' if ok else 'low_confidence'} "
                    f"(docs={len(evidences)}, max_dense={max_dense:.3f}, threshold={cfg.SIMILARITY_THRESHOLD})"
                )
            ],
        }

    def gerar_resposta(state: RAGState) -> dict[str, Any]:
        attempts = int(state.get("attempts", 0)) + 1
        feedback = state.get("correction_feedback")
        correction_block = (
            f"CORREÇÃO OBRIGATÓRIA DA TENTATIVA ANTERIOR:\n{feedback}"
            if feedback
            else ""
        )
        sys_prompt = SYNTHESIS_SYSTEM_PROMPT.format(correction_block=correction_block).strip()
        user_prompt = SYNTHESIS_USER_PROMPT.format(
            formatted_history=_format_chat_history(state.get("chat_history", [])),
            aggregate_stats=state.get("aggregate_stats") or "Sem estatísticas disponíveis.",
            formatted_evidence_xml_blocks=_format_evidence_xml(state.get("evidences", [])),
            sanitized_question=state["sanitized_question"],
        )
        response = active_llm.invoke(
            [
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        raw_content = getattr(response, "content", response)
        if isinstance(raw_content, list):
            answer_text = " ".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in raw_content
            ).strip()
        else:
            answer_text = str(raw_content).strip()

        return {
            "attempts": attempts,
            "answer": answer_text,
            "trilha": [f"gerar_resposta:attempt_{attempts}"],
        }

    def verificar_fidelidade(state: RAGState) -> dict[str, Any]:
        answer = (state.get("answer") or "").strip()
        evidences = state.get("evidences", [])
        retrieved_ids = {ev.review_id for ev in evidences}
        attempts = int(state.get("attempts", 1))

        if answer == VERBATIM_INSUFFICIENT_REFUSAL:
            return {
                "faithfulness_ok": True,
                "cited_review_ids": [],
                "status": "insufficient_evidence",
                "correction_feedback": None,
                "trilha": ["verificar_fidelidade:verbatim_refusal"],
            }

        raw_matches = _CITATION_RE.findall(answer)
        valid_citations: list[str] = []
        invalid_citations: list[str] = []
        for token in raw_matches:
            if token in retrieved_ids:
                if token not in valid_citations:
                    valid_citations.append(token)
            elif token not in invalid_citations:
                invalid_citations.append(token)

        if valid_citations and not invalid_citations:
            return {
                "faithfulness_ok": True,
                "cited_review_ids": valid_citations,
                "status": "answered",
                "correction_feedback": None,
                "trilha": [f"verificar_fidelidade:ok ({len(valid_citations)} citations)"],
            }

        if attempts < cfg.MAX_VERIFICATION_ATTEMPTS:
            reasons: list[str] = []
            if not valid_citations:
                reasons.append(
                    "Você não citou nenhum review_id válido de <evidence> no formato [review_id]."
                )
            if invalid_citations:
                reasons.append(
                    f"Os seguintes IDs citados não existem em <evidence>: {invalid_citations}. "
                    "Remova-os e cite apenas review_id presentes nas tags <evidence>."
                )
            return {
                "faithfulness_ok": False,
                "cited_review_ids": valid_citations,
                "correction_feedback": " ".join(reasons),
                "trilha": [f"verificar_fidelidade:retry (attempt={attempts})"],
            }

        # Safe fallback after exhausting MAX_VERIFICATION_ATTEMPTS
        top_refs = ", ".join(f"[{ev.review_id}]" for ev in evidences[:3])
        safe_answer = (
            f"{VERBATIM_INSUFFICIENT_REFUSAL}\n\n"
            f"Avaliações recuperadas para inspeção direta: {top_refs}"
            if top_refs
            else VERBATIM_INSUFFICIENT_REFUSAL
        )
        return {
            "faithfulness_ok": True,
            "answer": safe_answer,
            "cited_review_ids": [ev.review_id for ev in evidences[:3]],
            "status": "insufficient_evidence",
            "correction_feedback": None,
            "trilha": ["verificar_fidelidade:max_attempts_safe_fallback"],
        }

    def recusar(state: RAGState) -> dict[str, Any]:
        if state.get("guardrail_blocked"):
            msg = (
                "Sua consulta foi bloqueada pela política de segurança "
                "(possível tentativa de injeção de instruções ou payload suspeito)."
            )
            return {
                "answer": msg,
                "evidences": [],
                "cited_review_ids": [],
                "status": "blocked",
                "trilha": ["recusar:blocked"],
            }

        intent_dict = state.get("intent") or {}
        if intent_dict.get("intent_type") == "out_of_scope":
            msg = (
                "Esta pergunta está fora do escopo da base de avaliações de clientes da Olist "
                "(2016–2018). Posso ajudar com análises sobre pedidos, entregas, qualidade de "
                "produtos, notas e comentários de consumidores."
            )
            return {
                "answer": msg,
                "evidences": [],
                "cited_review_ids": [],
                "status": "out_of_scope",
                "trilha": ["recusar:out_of_scope"],
            }

        active_filters = _describe_active_filters(intent_dict)
        filter_note = (
            f"\n\nFiltros ativos aplicados: {active_filters}."
            if active_filters
            else ""
        )
        return {
            "answer": f"{VERBATIM_INSUFFICIENT_REFUSAL}{filter_note}",
            "cited_review_ids": [],
            "status": "insufficient_evidence",
            "trilha": ["recusar:insufficient_evidence"],
        }

    def route_after_guardrail(state: RAGState) -> str:
        return "recusar" if state.get("guardrail_blocked") else "rotear_consulta"

    def route_after_router(state: RAGState) -> str:
        intent = state.get("intent") or {}
        return "recusar" if intent.get("intent_type") == "out_of_scope" else "expandir_consulta"

    def route_after_confidence(state: RAGState) -> str:
        return "gerar_resposta" if state.get("confidence_ok") else "recusar"

    def route_after_verification(state: RAGState) -> str:
        if state.get("faithfulness_ok") or int(state.get("attempts", 0)) >= cfg.MAX_VERIFICATION_ATTEMPTS:
            return END
        return "gerar_resposta"

    graph = StateGraph(RAGState)
    for name, fn in [
        ("guardrail_entrada", guardrail_entrada),
        ("rotear_consulta", rotear_consulta),
        ("expandir_consulta", expandir_consulta),
        ("recuperar_hibrido", recuperar_hibrido),
        ("avaliar_evidencias", avaliar_evidencias),
        ("gerar_resposta", gerar_resposta),
        ("verificar_fidelidade", verificar_fidelidade),
        ("recusar", recusar),
    ]:
        graph.add_node(name, fn)

    graph.add_edge(START, "guardrail_entrada")
    graph.add_conditional_edges(
        "guardrail_entrada",
        route_after_guardrail,
        {"recusar": "recusar", "rotear_consulta": "rotear_consulta"},
    )
    graph.add_conditional_edges(
        "rotear_consulta",
        route_after_router,
        {"recusar": "recusar", "expandir_consulta": "expandir_consulta"},
    )
    graph.add_edge("expandir_consulta", "recuperar_hibrido")
    graph.add_edge("recuperar_hibrido", "avaliar_evidencias")
    graph.add_conditional_edges(
        "avaliar_evidencias",
        route_after_confidence,
        {"recusar": "recusar", "gerar_resposta": "gerar_resposta"},
    )
    graph.add_edge("gerar_resposta", "verificar_fidelidade")
    graph.add_conditional_edges(
        "verificar_fidelidade",
        route_after_verification,
        {"gerar_resposta": "gerar_resposta", END: END},
    )
    graph.add_edge("recusar", END)

    return graph.compile()
