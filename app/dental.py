"""Deterministic dental vocabulary: FDI tooth notation, anatomical sites and finding terms.

Both the rule-based dictation parser and the LLM grounding validator use this module, so
"is this finding supported by what the clinician said?" is always decided by the same
fixed vocabulary, whichever interpreter produced the finding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- FDI notation
QUAD_LABEL = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right",
              5: "upper right", 6: "upper left", 7: "lower left", 8: "lower right"}
PERMANENT_TYPES = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
                   5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar"}
PRIMARY_TYPES = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first molar", 5: "second molar"}
PERM_QUAD = {("upper", "right"): 1, ("upper", "left"): 2, ("lower", "left"): 3, ("lower", "right"): 4}
PRIM_QUAD = {("upper", "right"): 5, ("upper", "left"): 6, ("lower", "left"): 7, ("lower", "right"): 8}

_FDI_FULL = re.compile(r"^(?:[1-4][1-8]|[5-8][1-5])$")


def is_valid_fdi(code: object) -> bool:
    return code is not None and bool(_FDI_FULL.match(str(code)))


def fdi_label(code: str) -> str:
    q, p = int(code[0]), int(code[1])
    if q <= 4:
        return f"{QUAD_LABEL[q]} {PERMANENT_TYPES[p]}"
    return f"{QUAD_LABEL[q]} primary {PRIMARY_TYPES[p]}"


def site_display(fdi: str) -> str:
    return f"Tooth {fdi} ({fdi_label(fdi)})"


# ----------------------------------------------------------------------------- site parsing
_ARCH = r"(?P<arch>upper|maxillary|lower|mandibular)"
_SIDE = r"(?P<side>right|left)"
_PRIM = r"(?:(?P<prim>primary|deciduous|milk)\s+)?"
_TYPE = (r"(?P<type>central\s+incisors?|lateral\s+incisors?|canines?|cuspids?|"
         r"(?:first|1st|second|2nd)\s+pre-?molars?|(?:first|1st|second|2nd|third|3rd)\s+molars?|"
         r"wisdom\s+(?:tooth|teeth))\b")
_NUM = r"(?P<num>[1-8]|one|two|three|four|five|six|seven|eight)"
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}

_TOOTH_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("full", re.compile(rf"\b{_ARCH}\s+{_SIDE}\s+{_PRIM}{_TYPE}", re.I)),
    ("full", re.compile(rf"\b{_SIDE}\s+{_ARCH}\s+{_PRIM}{_TYPE}", re.I)),
    ("full", re.compile(rf"\b{_PRIM}{_TYPE}\s+(?:on|in|of)\s+(?:the\s+)?{_ARCH}\s+{_SIDE}(?:\s+side)?\b", re.I)),
    ("full", re.compile(rf"\b{_PRIM}{_TYPE}\s+(?:on|in|of)\s+(?:the\s+)?{_SIDE}\s+{_ARCH}\b", re.I)),
    ("num", re.compile(rf"\b{_ARCH}\s+{_SIDE}\s+{_NUM}\b(?!\s*(?:mm|cm)\b)(?![.,]\d)", re.I)),
    ("palmer", re.compile(r"\b(?P<code>[UL][RL])\s?(?P<num>[1-8])\b")),
    ("partial", re.compile(rf"\b{_ARCH}\s+{_PRIM}{_TYPE}", re.I)),
    ("partial", re.compile(rf"\b{_SIDE}\s+{_PRIM}{_TYPE}", re.I)),
    ("partial", re.compile(rf"\b{_PRIM}{_TYPE}", re.I)),
    # STT often writes tooth numbers as words: "four eight", "thirty-six", "tooth eighteen"
    ("spoken_pair", re.compile(
        r"\b(?P<w1>one|two|three|four|five|six|seven|eight)[\s-]+(?P<w2>one|two|three|four|five|six|seven|eight)\b"
        r"(?!\s*(?:mm|millimet\w*|cm|years?|yrs?|percent|days?|weeks?|months?)\b)", re.I)),
    ("spoken_num", re.compile(
        r"\b(?P<tens>eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|twenty|thirty|forty|fifty|"
        r"sixty|seventy|eighty)(?:[\s-]+(?P<unit>one|two|three|four|five|six|seven|eight))?\b"
        r"(?!\s*(?:mm|millimet\w*|cm|years?|yrs?|percent|days?|weeks?|months?)\b)", re.I)),
    ("tooth_pair", re.compile(
        r"\b(?:tooth|teeth|number)\s+(?P<d1>[1-8])\s*[-\u2013\s]\s*(?P<d2>[1-8])\b"
        r"(?!\s*(?:mm|millimet\w*|cm|years?|days?|weeks?|months?)\b)", re.I)),
    ("fdi", re.compile(
        r"(?<![\w./-])(?P<fdi>[1-8][1-8])(?![\w.]|-\d)"
        r"(?!\s*(?:mm|cm|years?|yrs?|y/?o|days?|weeks?|months?|degrees?|am|pm|hours?|minutes?)\b)(?!\s*[%°])",
        re.I)),
]

_REGION_PATTERNS = [
    re.compile(r"\b(?:(?:right|left)\s+)?maxillary\s+(?:sinus(?:es)?|antrum)\b", re.I),
    re.compile(r"\b(?:(?:right|left)\s+)?(?:tmj|temporomandibular\s+joint|condyle|condylar\s+head)\b", re.I),
    re.compile(r"\b(?:(?:right|left)\s+)?(?:anterior|posterior)\s+(?:maxilla|mandible)\b", re.I),
    re.compile(r"\b(?:lower|upper|mandibular|maxillary)\s+anteriors?(?:\s+region)?\b", re.I),
    re.compile(r"\b(?:(?:mandibular|maxillary)\s+)?(?:ramus|symphysis|tuberosity)\b", re.I),
]
_GENERAL_RE = re.compile(
    r"\bgenerali[sz]ed\b|\bfull[- ]mouth\b|\bthroughout\s+(?:the\s+)?(?:arch(?:es)?|dentition|mouth)\b|\ball\s+quadrants\b",
    re.I)
_BILATERAL_RE = re.compile(r"\b(?:both|bilateral(?:ly)?|all(?:\s+four)?)\b", re.I)


@dataclass
class SiteMatch:
    type: str            # tooth | region | generalized | unresolved
    label: str
    start: int
    end: int
    text: str
    fdi: str | None = None
    candidates: list[str] = field(default_factory=list)


def _arch(a: str) -> str:
    return "upper" if a.lower() in ("upper", "maxillary") else "lower"


def _position(type_text: str, primary: bool) -> int | None:
    t = type_text.lower()
    first = "first" in t or "1st" in t
    second = "second" in t or "2nd" in t
    if "central" in t:
        return 1
    if "lateral" in t:
        return 2
    if "canine" in t or "cuspid" in t:
        return 3
    if "premolar" in t or "pre-molar" in t:
        if primary:
            return None
        return 4 if first else 5 if second else None
    if "wisdom" in t or "third" in t or "3rd" in t:
        return None if primary else 8
    if "molar" in t:
        if primary:
            return 4 if first else 5 if second else None
        return 6 if first else 7 if second else None
    return None


def _fdi(arch: str, side: str, pos: int | None, primary: bool) -> str | None:
    if pos is None:
        return None
    q = (PRIM_QUAD if primary else PERM_QUAD)[(arch, side)]
    code = f"{q}{pos}"
    return code if is_valid_fdi(code) else None


def _candidates(arch: str | None, side: str | None, pos: int | None, primary: bool) -> list[str]:
    arches = [arch] if arch else ["upper", "lower"]
    sides = [side] if side else ["right", "left"]
    out = {c for a in arches for s in sides if (c := _fdi(a, s, pos, primary))}
    return sorted(out)


_TEENS = {"eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
          "eighteen": 18}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80}


def _spoken_code(gd: dict[str, str | None]) -> str | None:
    if gd.get("w1"):
        return f"{_NUM_WORDS[gd['w1'].lower()]}{_NUM_WORDS[gd['w2'].lower()]}"  # type: ignore[union-attr]
    tens = (gd.get("tens") or "").lower()
    if tens in _TEENS:
        return None if gd.get("unit") else str(_TEENS[tens])
    if tens in _TENS and gd.get("unit"):
        return str(_TENS[tens] + _NUM_WORDS[gd["unit"].lower()])  # type: ignore[union-attr]
    return None


def _overlaps(start: int, end: int, taken: list[tuple[int, int]]) -> bool:
    return any(start < e and end > s for s, e in taken)


def find_sites(text: str) -> list[SiteMatch]:
    """Find anatomical sites (teeth, regions, 'generalized') in a piece of text."""
    found: list[SiteMatch] = []
    taken: list[tuple[int, int]] = []
    bilateral = bool(_BILATERAL_RE.search(text))

    for kind, pat in _TOOTH_PATTERNS:
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end(), taken):
                continue
            gd = m.groupdict()
            primary = bool(gd.get("prim"))
            raw = m.group(0)
            if kind in ("spoken_pair", "spoken_num", "tooth_pair"):
                code = f"{gd['d1']}{gd['d2']}" if kind == "tooth_pair" else _spoken_code(gd)
                if not code or not is_valid_fdi(code):
                    continue
                found.append(SiteMatch("tooth", site_display(code), m.start(), m.end(), raw, fdi=code))
            elif kind == "fdi":
                code = gd["fdi"]
                if not is_valid_fdi(code):
                    continue
                found.append(SiteMatch("tooth", site_display(code), m.start(), m.end(), raw, fdi=code))
            elif kind == "palmer":
                code = gd["code"].upper()
                arch = "upper" if code[0] == "U" else "lower"
                side = "right" if code[1] == "R" else "left"
                fdi = _fdi(arch, side, int(gd["num"]), False)
                if not fdi:
                    continue
                found.append(SiteMatch("tooth", site_display(fdi), m.start(), m.end(), raw, fdi=fdi))
            elif kind == "num":
                n = gd["num"].lower()
                pos = _NUM_WORDS.get(n, None) or int(n)
                fdi = _fdi(_arch(gd["arch"]), gd["side"].lower(), pos, primary)
                if not fdi:
                    continue
                found.append(SiteMatch("tooth", site_display(fdi), m.start(), m.end(), raw, fdi=fdi))
            elif kind == "full":
                pos = _position(gd["type"], primary)
                fdi = _fdi(_arch(gd["arch"]), gd["side"].lower(), pos, primary)
                if not fdi:
                    continue
                found.append(SiteMatch("tooth", site_display(fdi), m.start(), m.end(), raw, fdi=fdi))
            else:  # partial: arch or side (or both) missing
                pos = _position(gd["type"], primary)
                if pos is None:
                    continue
                arch = _arch(gd["arch"]) if gd.get("arch") else None
                side = gd["side"].lower() if gd.get("side") else None
                cands = _candidates(arch, side, pos, primary)
                if not cands:
                    continue
                if bilateral and len(cands) > 1:
                    for c in cands:  # "both lower third molars" -> 38 and 48
                        found.append(SiteMatch("tooth", site_display(c), m.start(), m.end(), raw, fdi=c))
                else:
                    missing = [w for w, v in (("arch", arch), ("side", side)) if v is None]
                    label = f"{raw.strip()} ({' and '.join(missing)} not specified; could be {' / '.join(cands)})"
                    found.append(SiteMatch("unresolved", label, m.start(), m.end(), raw, candidates=cands))
            taken.append((m.start(), m.end()))

    for pat in _REGION_PATTERNS:
        for m in pat.finditer(text):
            if _overlaps(m.start(), m.end(), taken):
                continue
            label = re.sub(r"\s+", " ", m.group(0)).strip()
            found.append(SiteMatch("region", label[0].upper() + label[1:], m.start(), m.end(), m.group(0)))
            taken.append((m.start(), m.end()))

    for m in _GENERAL_RE.finditer(text):
        if not _overlaps(m.start(), m.end(), taken):
            found.append(SiteMatch("generalized", "Generalized", m.start(), m.end(), m.group(0)))
            taken.append((m.start(), m.end()))

    found.sort(key=lambda s: s.start)
    return found


# ----------------------------------------------------------------------------- findings
@dataclass(frozen=True)
class FindingDef:
    key: str
    label: str
    pattern: re.Pattern[str]
    qualifiers: tuple[tuple[str, re.Pattern[str]], ...] = ()
    interpretive: bool = False                 # diagnostic-level term: always flag for review
    suppressed_by: tuple[str, ...] = ()        # drop when a more specific finding matched


def _p(rx: str) -> re.Pattern[str]:
    return re.compile(rx, re.I)


def _q(*pairs: tuple[str, str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple((name, _p(rx)) for name, rx in pairs)


FINDINGS: list[FindingDef] = [
    FindingDef("rct", "Root canal treated",
               _p(r"\broot\s+canal\s+(?:treated|treatment|therapy|filled|filling|obturat\w*)\b|\brct\b|"
                  r"\bendodontically\s+treated\b|\bobturat(?:ed|ion)\b"),
               _q(("adequate obturation", r"\badequate(?:ly)?\b"),
                  ("short obturation", r"\bshort\b|\bunder-?fill\w*|\binadequate(?:ly)?\b"),
                  ("overextended", r"\bover-?extend\w*|\bover-?fill\w*"))),
    FindingDef("impaction", "Impaction",
               _p(r"\bimpact(?:ed|ion)\b"),
               _q(("horizontal", r"\bhorizontal(?:ly)?\b"),
                  ("mesioangular", r"\bmesio-?\s?angular(?:ly)?\b"),
                  ("distoangular", r"\bdisto-?\s?angular(?:ly)?\b"),
                  ("vertical", r"\bvertical(?:ly)?\b"),
                  ("buccoangular", r"\bbucco-?\s?angular(?:ly)?\b"),
                  ("inverted", r"\binverted\b"),
                  ("partially erupted", r"\bpartial(?:ly)?\s+erupted\b|\bpartial(?:ly)?\b"),
                  ("complete bony", r"\b(?:complete(?:ly)?|full(?:y)?)\s+bony\b"))),
    FindingDef("periapical", "Periapical radiolucency",
               _p(r"\b(?:peri-?apical|apical)\s+(?:radiolucen\w*|lesion|rarefaction|pathology|abscess|area|changes?)\b|"
                  r"\bradiolucen\w*\s+(?:at|around|near|involving)\s+(?:the\s+)?(?:apex|apices|apical\s+region|root\s+apex)\b|"
                  r"\bpa\s+(?:lesion|radiolucency)\b"),
               _q(("well-defined", r"\bwell[- ]defined\b|\bwell[- ]circumscribed\b|\bcorticated\b"),
                  ("ill-defined", r"\bill[- ]defined\b|\bdiffuse\b"))),
    FindingDef("caries", "Caries",
               _p(r"\bcaries\b|\bcarious\b|\bdecay(?:ed)?\b|\bcavit(?:y|ies)\b"),
               _q(("deep", r"\bdeep\b"),
                  ("secondary", r"\bsecondary\b|\brecurrent\b"),
                  ("proximal", r"\bproximal\b|\binterproximal\b"),
                  ("mesial", r"\bmesial(?:ly)?\b"),
                  ("distal", r"\bdistal(?:ly)?\b"),
                  ("occlusal", r"\bocclusal\b"),
                  ("incipient", r"\bincipient\b|\benamel\s+only\b|\blimited\s+to\s+enamel\b"),
                  ("approaching pulp", r"\bapproach\w*\s+(?:the\s+)?pulp|\bclose\s+to\s+(?:the\s+)?pulp|\bnear\s+(?:the\s+)?pulp"),
                  ("involving pulp", r"\binvolv\w*\s+(?:the\s+)?pulp\b|\bpulpal\s+involvement\b"))),
    FindingDef("bone_loss", "Alveolar bone loss",
               _p(r"\b(?:alveolar\s+)?bone\s+(?:loss|resorption)\b|\balveolar\s+resorption\b|\bcrestal\s+bone\s+loss\b"),
               _q(("horizontal", r"\bhorizontal\b"), ("vertical", r"\bvertical\b|\bangular\b"),
                  ("mild", r"\bmild\b"), ("moderate", r"\bmoderate\b"), ("severe", r"\bsevere\b"),
                  ("localized", r"\blocali[sz]ed\b"))),
    FindingDef("canal_proximity", "Root proximity to mandibular canal",
               _p(r"\b(?:close|closely|proxim\w*|approximat\w*|contact\w*|abutt\w*|overlap\w*|superimpos\w*|"
                  r"touch\w*|encroach\w*|near|related|relation)\b[^.;]{0,40}?"
                  r"\b(?:mandibular\s+canal|inferior\s+alveolar(?:\s+nerve)?(?:\s+canal)?|ian(?:\s+canal)?|alveolar\s+canal)\b|"
                  r"\b(?:mandibular\s+canal|inferior\s+alveolar\s+(?:nerve\s+)?canal|ian\s+canal)\b[^.;]{0,40}?"
                  r"\b(?:close|proxim\w*|contact\w*|abutt\w*|overlap\w*|superimpos\w*|touch\w*|near|groov\w*|"
                  r"interrupt\w*|divert\w*|diversion|darken\w*|narrow\w*)"),
               _q(("cortical interruption", r"\binterrupt\w*|\bloss\s+of\s+(?:the\s+)?(?:white|cortical)\s+line\b"),
                  ("canal diversion", r"\bdivert\w*|\bdiversion\b"),
                  ("root darkening", r"\bdarken\w*"))),
    FindingDef("missing", "Missing tooth",
               _p(r"\bmissing\b|\bedentulous\b|\bextracted\b|\bextraction\s+socket\b")),
    FindingDef("restoration", "Restoration",
               _p(r"\brestor(?:ed|ation|ations)\b|(?<!canal\s)\bfill(?:ing|ings|ed)\b|\bamalgam\b|\bcomposite\b|\bgic\b|"
                  r"\binlay\b|\bonlay\b"),
               _q(("amalgam", r"\bamalgam\b"), ("composite", r"\bcomposite\b"), ("GIC", r"\bgic\b"),
                  ("overhanging", r"\boverhang\w*"),
                  ("defective margins", r"\bdefective\b|\bleaking\b|\bmarginal\s+(?:gap|breakdown|defect)\b"))),
    FindingDef("crown", "Crown",
               _p(r"\bcrowned\b|\b(?:pfm|zirconia|metal|ceramic|porcelain|metal-ceramic|full[- ]coverage|prosthetic)\s+crowns?\b|"
                  r"\bcrowns?\s+(?:present|in\s+place|placed|noted)\b")),
    FindingDef("implant", "Implant",
               _p(r"\bimplants?\b"),
               _q(("peri-implant bone loss", r"\bperi-?implant\b"), ("osseointegrated", r"\bosseo-?integrat\w*"))),
    FindingDef("root_stump", "Residual root",
               _p(r"\broot\s+stumps?\b|\bresidual\s+roots?\b|\bretained\s+roots?\b|\broot\s+(?:pieces?|remnants?|tips?)\b")),
    FindingDef("root_resorption", "Root resorption",
               _p(r"\broot\s+resorption\b|\b(?:external|internal)\s+(?:root\s+)?resorption\b|\bresorption\s+of\s+(?:the\s+)?\w*\s*root\b")),
    FindingDef("fracture", "Fracture",
               _p(r"\bfractur(?:e|ed|es)\b"),
               _q(("root", r"\broot\b"), ("crown", r"\bcrown\b"), ("vertical", r"\bvertical\b"),
                  ("horizontal", r"\bhorizontal\b"))),
    FindingDef("calculus", "Calculus",
               _p(r"\bcalculus\b|\btartar\b|\bcalcul(?:ar|us)\s+deposits?\b"),
               _q(("subgingival", r"\bsub-?gingival\b"), ("supragingival", r"\bsupra-?gingival\b"))),
    FindingDef("pdl_widening", "PDL space widening",
               _p(r"\bwiden\w*\s+(?:of\s+)?(?:the\s+)?(?:pdl|periodontal\s+ligament)(?:\s+space)?\b|"
                  r"\b(?:pdl|periodontal\s+ligament)(?:\s+space)?\s+(?:is\s+|appears\s+)?widen\w*")),
    FindingDef("sinus_thickening", "Sinus mucosal thickening",
               _p(r"\bmucosal\s+thickening\b")),
    FindingDef("sinus_opacification", "Sinus opacification",
               _p(r"\b(?:sinus|antral)\s+(?:opacification|opacity|haziness|hazy)\b|\bopacification\s+of\s+(?:the\s+)?\w*\s*(?:sinus|antrum)\b")),
    FindingDef("retained_deciduous", "Retained primary tooth",
               _p(r"\bretained\s+(?:deciduous|primary|milk)\b|\bover-?retained\b")),
    FindingDef("supernumerary", "Supernumerary tooth",
               _p(r"\bsupernumerary\b|\bmesiodens\b")),
    FindingDef("unerupted", "Unerupted / developing tooth",
               _p(r"\bun-?erupted\b|\bdeveloping\b|\btooth\s+bud\b|\bnot\s+(?:yet\s+)?erupted\b")),
    FindingDef("cyst", "Cystic lesion",
               _p(r"\bcyst(?:s|ic)?\b"),
               interpretive=True),
    FindingDef("radiolucency", "Radiolucency (non-specific)",
               _p(r"\bradiolucen(?:t|cy|cies)\b"),
               suppressed_by=("periapical", "caries", "pdl_widening")),
    FindingDef("radiopacity", "Radiopacity (non-specific)",
               _p(r"\bradio-?opa(?:que|city|cities)\b"),
               suppressed_by=("restoration", "rct", "calculus", "implant", "crown")),
]
FINDING_BY_KEY: dict[str, FindingDef] = {d.key: d for d in FINDINGS}
CATEGORY_LABELS: dict[str, str] = {d.key: d.label for d in FINDINGS} | {"other": "Other finding"}

# Extra words people use in questions ("any lesions?", "what about the nerve?").
_QUESTION_SYNONYMS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (_p(r"\blesions?\b|\bpatholog\w*"), ("periapical", "radiolucency", "cyst")),
    (_p(r"\binfection\b|\babscess\b"), ("periapical",)),
    (_p(r"\broot\s+canals?\b|\bendo\b"), ("rct",)),
    (_p(r"\bnerve\b|\bian\b|\bmandibular\s+canal\b"), ("canal_proximity",)),
    (_p(r"\bsinus\b|\bantrum\b"), ("sinus_thickening", "sinus_opacification")),
    (_p(r"\bfillings?\b|\brestorations?\b"), ("restoration",)),
    (_p(r"\bperiodont\w*|\bgums?\b"), ("bone_loss", "calculus")),
    (_p(r"\bbone\s+(?:levels?|height|support)\b"), ("bone_loss",)),
    (_p(r"\bcrowns?\b"), ("crown",)),
    (_p(r"\bimpact\w*|\bwisdom\b"), ("impaction",)),
    (_p(r"\bradiolucen\w*"), ("radiolucency", "periapical")),
]


@dataclass
class FindingMatch:
    defn: FindingDef
    start: int
    end: int
    text: str


def find_findings(text: str) -> list[FindingMatch]:
    """All vocabulary findings in a piece of text, one per category, ordered by position."""
    matches = [FindingMatch(d, m.start(), m.end(), m.group(0)) for d in FINDINGS for m in d.pattern.finditer(text)]
    keys = {m.defn.key for m in matches}
    matches = [m for m in matches if not any(k in keys for k in m.defn.suppressed_by)]
    out: list[FindingMatch] = []
    seen: set[str] = set()
    for m in sorted(matches, key=lambda x: x.start):
        if m.defn.key not in seen:
            seen.add(m.defn.key)
            out.append(m)
    return out


def qualifiers_in(defn: FindingDef, text: str) -> list[str]:
    quals = [name for name, rx in defn.qualifiers if rx.search(text)]
    if _NO_CHANGE_RE.search(text) or re.search(r"\bunchanged\b|\bstable\b", text, re.I):
        quals.append("unchanged")
    return quals


def question_term_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of question-only vocabulary ("lesions", "nerve", "wisdom"), so it is never read as a name."""
    return [m.span() for rx, _ in _QUESTION_SYNONYMS for m in rx.finditer(text)]


def categories_in_question(text: str) -> set[str]:
    cats = {m.defn.key for m in find_findings(text)}
    for rx, keys in _QUESTION_SYNONYMS:
        if rx.search(text):
            cats.update(keys)
    return cats


def evidence_supports(category: str, evidence: str) -> bool:
    """True when the evidence text contains a term from this category's vocabulary."""
    defn = FINDING_BY_KEY.get(category)
    if defn is None:
        return False
    return bool(defn.pattern.search(evidence))


# ----------------------------------------------------------------------------- assertion status
_UNCERTAIN_RE = re.compile(
    r"\b(?:possibl[ey]|probabl[ey]|likely|suspect(?:ed)?|suspicious(?:\s+of)?|suggestive\s+of|suggests?|query|"
    r"questionable|appears?(?:\s+to\s+be)?|seems?(?:\s+to\s+be)?|may\s+be|might\s+be|could\s+be|"
    r"cannot\s+(?:be\s+)?rule[ds]?\s+out|cannot\s+be\s+ruled\s+out|can'?t\s+rule\s+out|cannot\s+(?:be\s+)?exclude[d]?|"
    r"not\s+sure|not\s+certain|doubtful|equivocal|indeterminate|rule\s+out|r/o|\?)", re.I)
_NO_CHANGE_RE = re.compile(r"\bno\s+(?:significant\s+|interval\s+|appreciable\s+)?change\b|\bnot\s+changed\b", re.I)
_NEG_BEFORE_RE = re.compile(r"\b(?:no|not|without|absence\s+of|absent|negative\s+for|free\s+of|nil|none|neither|nor)\b", re.I)
_NEG_AFTER_RE = re.compile(
    r"^\W*(?:\w+\W+){0,4}?(?:is\s+|are\s+)?(?:not\s+(?:seen|noted|observed|evident|present|detected|visible|identified|appreciated)|"
    r"absent|ruled\s+out)\b", re.I)


def _mask(text: str, rx: re.Pattern[str]) -> str:
    return rx.sub(lambda m: " " * len(m.group(0)), text)


def assertion_status(piece: str, start: int, end: int, sentence_is_question: bool = False) -> tuple[bool, str, str | None]:
    """Return (present, certainty, uncertainty cue) for the finding at piece[start:end]."""
    unc = _UNCERTAIN_RE.search(piece)
    cue = unc.group(0).strip() if unc else ("?" if sentence_is_question else None)
    masked = _mask(_mask(piece, _NO_CHANGE_RE), _UNCERTAIN_RE)
    window = " ".join(re.findall(r"\S+", masked[:start])[-6:])
    negated = bool(_NEG_BEFORE_RE.search(window)) or bool(_NEG_AFTER_RE.search(masked[end:]))
    return (not negated, "uncertain" if cue else "stated", cue)
