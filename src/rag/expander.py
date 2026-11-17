"""Portuguese multi-query expansion for improved semantic and lexical recall."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.rag.prompts import MULTI_QUERY_PROMPT

_PREFIX_CLEAN_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)\]])\s*")


def normalizar_variacoes(
    original_query: str,
    raw_variations: list[str],
    max_variations: int = 2,
) -> list[str]:
    """Clean, deduplicate, and prepend the base query to search variations."""
    base = original_query.strip()
    queries: list[str] = [base] if base else []
    seen: set[str] = {base.lower()} if base else set()

    for item in raw_variations:
        cleaned = _PREFIX_CLEAN_RE.sub("", str(item)).strip().strip('"').strip("'")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) >= (1 if base else 0) + max_variations:
            break

    return queries


def _extrair_variacoes(raw_text: str, original_query: str, max_variations: int = 2) -> list[str]:
    """Parse line-delimited LLM variations, clean prefixes, and deduplicate."""
    return normalizar_variacoes(
        original_query,
        raw_text.splitlines(),
        max_variations=max_variations,
    )


def expand_query(semantic_query: str, llm: Any, qtd: int = 2) -> list[str]:
    """Generate colloquial Brazilian Portuguese variations alongside the original query."""
    prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
    chain = prompt | llm | StrOutputParser()
    raw_output = chain.invoke({"semantic_query": semantic_query, "qtd": qtd})
    return _extrair_variacoes(raw_output, semantic_query, max_variations=qtd)
