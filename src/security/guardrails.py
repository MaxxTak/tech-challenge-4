"""Defense-in-depth LGPD PII masking, prompt injection detection, and secret scrubbing."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from typing import Any

# Regex patterns ordered from more specific formats to broader ones
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_CPF_RE = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CEP_RE = re.compile(r"\b\d{5}-\d{3}\b")
_PHONE_RE = re.compile(
    r"(?:\+55[\s-]?(?:\(?\d{2}\)?[\s-]?)?(?:9\d{4}|\d{4})-?\d{4}\b"
    r"|\(\d{2}\)\s*(?:9\d{4}|\d{4})-?\d{4}\b"
    r"|\b\d{2}\s+(?:9\d{4}|\d{4})-?\d{4}\b"
    r"|\b\d{2}(?:9\d{4}|\d{4})-\d{4}\b)"
)

_BASE64_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]{40,}")

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore_previous_instructions", re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE)),
    ("ignore_instrucoes_pt", re.compile(r"ignore\s+(todas\s+as\s+)?instru[çc][õo]es\s+(anteriores|acima)", re.IGNORECASE)),
    ("desconsidere_regras_pt", re.compile(r"(desconsidere|esque[çc]a)\s+(todas\s+as\s+)?(regras|instru[çc][õo]es|restri[çc][õo]es)", re.IGNORECASE)),
    ("system_prompt_exfil_en", re.compile(r"(print|reveal|show|dump|repeat|output)\s+(your\s+|the\s+)?(system\s+prompt|instructions|hidden\s+prompt)", re.IGNORECASE)),
    ("system_prompt_exfil_pt", re.compile(r"(imprima|mostre|revele|exiba|repita)\s+(o\s+|seu\s+|a\s+)?(system\s+prompt|prompt\s+(do\s+)?sistema|instru[çc][õo]es\s+internas)", re.IGNORECASE)),
    ("role_override_en", re.compile(r"\b(you\s+are\s+now|act\s+as\s+DAN|jailbreak|developer\s+mode)\b", re.IGNORECASE)),
    ("role_override_pt", re.compile(r"\b(agora\s+voc[êe]\s+[ée]|finja\s+ser|atue\s+(?:em|como)\s+(?:modo\s+)?DAN|modo\s+(?:desenvolvedor|DAN)|sem\s+restri[çc][õo]es)\b", re.IGNORECASE)),
    ("secret_exfil", re.compile(r"(GOOGLE_API_KEY|OPENAI_API_KEY|RAG_API_KEY|os\.environ|process\.env|\.env\b)", re.IGNORECASE)),
    ("xml_tag_spoof", re.compile(r"</?\s*(system|evidence|metadata_stats|assistant)\s*>", re.IGNORECASE)),
    ("command_exec", re.compile(r"\b(__import__|subprocess|eval\(|exec\(|rm\s+-rf|curl\s+http)\b", re.IGNORECASE)),
]

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"sk-(?:proj-)?[0-9A-Za-z_-]{16,}"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._-]+"), r"\1[REDACTED_TOKEN]"),
    (re.compile(r"(?i)\b(api_key|key|token|password|secret)\s*=\s*[^\s&\"']+"), r"\1=[REDACTED]"),
]


def validar_cpf(cpf_str: str) -> bool:
    """Validate Brazilian CPF using official mod-11 check digits."""
    digits = re.sub(r"\D", "", cpf_str)
    if len(digits) != 11 or len(set(digits)) == 1:
        return False

    for check_idx in (9, 10):
        total = sum(int(digits[i]) * ((check_idx + 1) - i) for i in range(check_idx))
        expected = (total * 10) % 11
        expected = 0 if expected == 10 else expected
        if int(digits[check_idx]) != expected:
            return False
    return True


def validar_luhn(numero_str: str) -> bool:
    """Validate payment card numbers using the Luhn mod-10 checksum."""
    digits = re.sub(r"\D", "", numero_str)
    if not (13 <= len(digits) <= 19) or len(set(digits)) == 1:
        return False

    total = 0
    reverse_digits = digits[::-1]
    for idx, ch in enumerate(reverse_digits):
        val = int(ch)
        if idx % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


def hash_dado(valor: str) -> str:
    """Return deterministic 12-character SHA-256 hex prefix for LGPD auditing."""
    return hashlib.sha256(valor.encode("utf-8")).hexdigest()[:12]


def _entropia(texto: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not texto:
        return 0.0
    counts = Counter(texto)
    length = len(texto)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def detectar_pii(texto: str) -> list[dict[str, Any]]:
    """Scan text for PII entities and flag checksum validity."""
    if not texto:
        return []

    findings: list[dict[str, Any]] = []
    occupied_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in occupied_spans)

    detectors: list[tuple[str, re.Pattern[str], Any]] = [
        ("email", _EMAIL_RE, lambda _v: True),
        ("cartao", _CARD_RE, validar_luhn),
        ("cpf", _CPF_RE, validar_cpf),
        ("cep", _CEP_RE, lambda _v: True),
        ("telefone", _PHONE_RE, lambda v: 10 <= len(re.sub(r"\D", "", v)) <= 13),
    ]

    for pii_type, pattern, validator in detectors:
        for match in pattern.finditer(texto):
            start, end = match.span()
            if _overlaps(start, end):
                continue
            raw_val = match.group(0)
            is_valid = bool(validator(raw_val))
            findings.append(
                {
                    "tipo": pii_type,
                    "valor": raw_val,
                    "span": (start, end),
                    "valido": is_valid,
                }
            )
            if is_valid:
                occupied_spans.append((start, end))

    findings.sort(key=lambda item: item["span"][0])
    return findings


def mascarar(texto: str) -> tuple[str, list[str]]:
    """Mask validated PII entities with ***<last2> and return (masked_text, sha256_hashes)."""
    if not texto:
        return texto, []

    valid_findings = [item for item in detectar_pii(texto) if item["valido"]]
    if not valid_findings:
        return texto, []

    masked = texto
    hashes: list[str] = []
    seen_hashes: set[str] = set()

    for item in reversed(valid_findings):
        start, end = item["span"]
        raw_val = item["valor"]
        suffix = raw_val[-2:] if len(raw_val) >= 2 else raw_val
        masked = f"{masked[:start]}***{suffix}{masked[end:]}"
        digest = hash_dado(raw_val)
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            hashes.append(digest)

    hashes.reverse()
    return masked, hashes


def detectar_injection(texto: str) -> dict[str, Any]:
    """Detect direct prompt injection patterns and high-entropy base64-like payloads."""
    if not texto or not texto.strip():
        return {"blocked": False, "score": 0, "motivos": []}

    normalized = unicodedata.normalize("NFKC", texto)
    motivos: list[str] = []

    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            motivos.append(f"pattern:{label}")

    for match in _BASE64_TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if _entropia(token) > 4.0:
            motivos.append("high_entropy_payload")
            break

    score = len(motivos)
    return {
        "blocked": score > 0,
        "score": score,
        "motivos": motivos,
    }


def mascarar_segredos(texto: str) -> str:
    """Scrub API keys, bearer tokens, and secret parameters from log messages."""
    if not texto:
        return texto
    scrubbed = str(texto)
    for pattern, replacement in _SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed
