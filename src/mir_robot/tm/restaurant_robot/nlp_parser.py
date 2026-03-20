from dataclasses import dataclass
from typing import Optional
import re


VI_NUMBER_WORDS = {
    "mot": 1, "một": 1,
    "hai": 2,
    "ba": 3,
    "bon": 4, "bốn": 4, "tu": 4, "tư": 4,
    "nam": 5, "năm": 5, "lam": 5, "lăm": 5,
    "sau": 6, "sáu": 6,
    "bay": 7, "bảy": 7,
    "tam": 8, "tám": 8,
    "chin": 9, "chín": 9,
    "muoi": 10, "mười": 10,
}

VI_NUMBER_PHRASES = {
    "mười một": 11, "muoi mot": 11,
    "mười hai": 12, "muoi hai": 12,
    "mười ba": 13, "muoi ba": 13,
    "mười bốn": 14, "muoi bon": 14,
    "mười lăm": 15, "muoi lam": 15,
    "mười sáu": 16, "muoi sau": 16,
    "mười bảy": 17, "muoi bay": 17,
    "mười tám": 18, "muoi tam": 18,
    "mười chín": 19, "muoi chin": 19,
}


@dataclass
class ParseResult:
    intent: str = "UNKNOWN"
    raw_text: str = ""
    table: Optional[int] = None
    qty: int = 1


def _to_number(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return VI_NUMBER_WORDS.get(token)


def _extract_table_and_qty(cleaned: str):
    tokens = re.findall(r"\d+|[\wÀ-ỹ]+", cleaned.lower())
    table = None
    qty = 1

    for phrase, value in VI_NUMBER_PHRASES.items():
        if phrase in cleaned:
            if table is None and ("bàn" in cleaned or "ban" in cleaned):
                table = value
            qty = value

    for idx, token in enumerate(tokens):
        value = _to_number(token)
        if value is not None:
            qty = value
            if idx > 0 and tokens[idx - 1] in ("bàn", "ban", "so", "số"):
                table = value

    if table is None and ("bàn" in cleaned or "ban" in cleaned):
        for token in tokens:
            value = _to_number(token)
            if value is not None:
                table = value
                break

    return table, qty


def parse(text: str) -> ParseResult:
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return ParseResult(intent="EMPTY", raw_text=text)

    table, qty = _extract_table_and_qty(cleaned)

    if "về" in cleaned or "ve" in cleaned:
        return ParseResult(intent="GO_HOME", raw_text=text, table=table, qty=qty)
    if "hủy" in cleaned or "huy" in cleaned:
        return ParseResult(intent="CANCEL", raw_text=text, table=table, qty=qty)
    if "thanh toán" in cleaned or "thanh toan" in cleaned:
        return ParseResult(intent="PAY", raw_text=text, table=table, qty=qty)
    if "ok" in cleaned or "xác nhận" in cleaned or "xac nhan" in cleaned:
        return ParseResult(intent="CONFIRM", raw_text=text, table=table, qty=qty)
    if "bàn" in cleaned or "ban" in cleaned:
        return ParseResult(intent="GO_TABLE", raw_text=text, table=table, qty=qty)
    if "nước" in cleaned or "nuoc" in cleaned or "ly" in cleaned:
        return ParseResult(intent="ORDER", raw_text=text, table=table, qty=qty)

    return ParseResult(intent="UNKNOWN", raw_text=text, table=table, qty=qty)
