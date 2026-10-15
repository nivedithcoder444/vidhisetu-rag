import re
from typing import Optional


def matches_act_filter(
    filter_str: Optional[str],
    act: str,
    doc_id: str,
    source: str,
    chunk_id: str = ""
) -> bool:
    """
    Determines whether a legal chunk matches the requested Act filter or hint.
    Prevents short abbreviation collision (e.g. 'bns' must not match 'bnss').
    Supports canonical Act titles and standard legal aliases.
    """
    if not filter_str:
        return True

    f = filter_str.lower().strip()
    a = act.lower().strip()
    d = doc_id.lower().strip()
    s = source.lower().strip()
    c = chunk_id.lower().strip()

    # 1. Whole-word match for short abbreviations
    # \bbns\b matches "bns" or "BNS-S105-C001" or "bns.pdf", but NEVER "bnss"
    if re.search(rf'\b{re.escape(f)}\b', a) or re.search(rf'\b{re.escape(f)}\b', d) or re.search(rf'\b{re.escape(f)}\b', s):
        return True
    if c and (c.startswith(f"{f}-") or re.search(rf'\b{re.escape(f)}\b', c)):
        return True

    # 2. Known statutory aliases
    aliases = {
        "bns": ["bharatiya nyaya", "nyaya sanhita"],
        "bnss": ["bharatiya nagarik", "nagarik suraksha"],
        "bsa": ["bharatiya sakshya", "sakshya adhiniyam"],
        "cpc": ["code of civil procedure", "civil procedure"],
        "crpc": ["code of criminal procedure", "criminal procedure 1973"],
        "ipc": ["indian penal code"],
        "contract": ["indian contract act", "contract act"],
        "arms": ["arms act"],
        "constitution": ["constitution of india"],
        "pmla": ["prevention of money laundering", "money laundering"],
        "ndps": ["narcotic drugs", "narcotics act"]
    }

    for alias_key, alias_phrases in aliases.items():
        # If filter requests this alias (e.g. filter is 'bns' or 'bharatiya nyaya sanhita')
        if f == alias_key or any(p in f for p in alias_phrases):
            if d == alias_key or any(p in a for p in alias_phrases) or any(p in s for p in alias_phrases):
                return True

    # 3. Substring match fallback for non-ambiguous multi-word titles
    short_abbrevs = {"bns", "bnss", "bsa", "cpc", "crpc", "ipc", "nia"}
    if f not in short_abbrevs and (f in a or f in d or f in s):
        return True

    return False
