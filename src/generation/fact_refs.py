from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Set, List

from src.domain.contracts import FactCard


_FACT_REF_RE = re.compile(r"\[(A\d+-F\d+)\]")


@dataclass(frozen=True)
class FactRefCheck:
    ok: bool
    used: Set[str]
    unknown: Set[str]


def collect_known_fact_ids(fact_cards: List[FactCard]) -> Set[str]:
    known: Set[str] = set()
    for card in fact_cards:
        for f in card.facts:
            known.add(f.fact_id)
    return known


def check_fact_refs(script: str, fact_cards: List[FactCard]) -> FactRefCheck:
    used = set(_FACT_REF_RE.findall(script or ""))
    known = collect_known_fact_ids(fact_cards)
    unknown = {fid for fid in used if fid not in known}
    return FactRefCheck(ok=len(unknown) == 0, used=used, unknown=unknown)
