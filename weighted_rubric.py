"""Load rubrics as weighted criteria (no multi-level score columns)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from type import RKTRoot


@dataclass
class WeightedCriterion:
    """One top-level criterion: display name, numeric weight, rubric text for extraction."""

    name: str
    weight: float
    body: str


def format_for_skill_extract(rows: list[WeightedCriterion]) -> str:
    """Single document for skill extraction: headers preserve category + weight."""
    parts: list[str] = []
    for r in rows:
        parts.append(f"### {r.name}\nWeight: {r.weight}\n\n{r.body}\n")
    return "\n".join(parts).strip()


def format_categories_for_tree(rows: list[WeightedCriterion]) -> str:
    """Ordered category list the skill-tree model must mirror (one rubric line each)."""
    lines = []
    for i, r in enumerate(rows, start=1):
        lines.append(f'{i}. Title (use EXACTLY as rubric line "description"): "{r.name}" — points: {r.weight}')
    return "\n".join(lines)


def _norm_key(s: str) -> str:
    return " ".join(s.strip().lower().split())


def attach_weights_from_rows(
    root: RKTRoot,
    rows: list[WeightedCriterion],
    *,
    sync_descriptions: bool = True,
) -> RKTRoot:
    """
    Set each rubric line's weight from the source list (by index, then by name match).
    Weights are stored as non-negative (absolute value of source stake / points).
    When sync_descriptions is True and row counts match, copy category title from the source file
    so rubric line descriptions stay descriptive (never placeholders like "---").
    """
    by_name = {_norm_key(r.name): r.weight for r in rows}
    new_rows = []
    same_len = sync_descriptions and len(root.rows) == len(rows)
    for i, line in enumerate(root.rows):
        w = line.weight
        desc = line.description
        if i < len(rows):
            w = rows[i].weight
            if same_len:
                nm = (rows[i].name or "").strip()
                if nm and not _is_placeholder_title(nm):
                    desc = nm
        else:
            k = _norm_key(line.description)
            if k in by_name:
                w = by_name[k]
        if w is not None:
            w = abs(float(w))
        new_rows.append(line.model_copy(update={"weight": w, "description": desc}))
    return root.model_copy(update={"rows": new_rows})


def _is_placeholder_title(s: str) -> bool:
    t = s.strip()
    return not t or t in ("---", "—", "-", "...", "…") or re.match(r"^[\-–_=]{2,}\s*$", t) is not None


def merge_consecutive_same_weight(rows: list[WeightedCriterion]) -> list[WeightedCriterion]:
    """
    Merge adjacent criteria with the same nominal weight into one row (bodies concatenated;
    weights summed). Use when a rubric repeats the same point value for related parts (e.g.
    four 10-point Excel tasks) and you want one rubric line with multiple leaves.
    """
    if not rows:
        return []
    out: list[WeightedCriterion] = []
    acc = rows[0]
    for r in rows[1:]:
        if r.weight == acc.weight:
            acc = WeightedCriterion(
                name=acc.name,
                weight=acc.weight + r.weight,
                body=(acc.body.rstrip() + "\n\n" + r.body.strip()).strip(),
            )
        else:
            out.append(acc)
            acc = r
    out.append(acc)
    return out


def normalize_rkt_positive_weights(root: RKTRoot) -> RKTRoot:
    """Ensure all weights on rubric lines and nested rules are non-negative (use abs)."""

    def norm_rule(rule):
        t = getattr(rule, "type", None)
        if t == "simple rule":
            w = getattr(rule, "weight", None)
            if w is not None and w < 0:
                return rule.model_copy(update={"weight": abs(float(w))})
            return rule
        children = [norm_rule(c) for c in (getattr(rule, "children", None) or [])]
        bw = getattr(rule, "weight", None)
        nb = abs(float(bw)) if bw is not None and bw < 0 else bw
        return rule.model_copy(update={"weight": nb, "children": children})

    new_rows = []
    for row in root.rows:
        rw = getattr(row, "weight", None)
        nrw = abs(float(rw)) if rw is not None and rw < 0 else rw
        desc = (row.description or "").strip()
        if _is_placeholder_title(desc):
            desc = "Rubric category"
        new_rows.append(
            row.model_copy(
                update={
                    "weight": nrw,
                    "description": desc,
                    "children": [norm_rule(c) for c in (row.children or [])],
                }
            )
        )
    return root.model_copy(update={"rows": new_rows})


def load_weighted_criteria_csv(
    path: str | Path,
    *,
    name_column: str = "Criteria",
    weight_column: str = "Weight",
    body_column: str | None = None,
) -> list[WeightedCriterion]:
    """
    Load a CSV with criterion names, weights, and a single description column
    (e.g. open-ended-response.csv: use Score 4 as body via body_column).
    """
    path = Path(path)
    df = pd.read_csv(path)
    if body_column is None:
        candidates = [c for c in df.columns if c not in (name_column, weight_column)]
        if not candidates:
            raise ValueError(f"No description column found in {path}")
        body_column = candidates[0]
    for col in (name_column, weight_column, body_column):
        if col not in df.columns:
            raise ValueError(f"Missing column {col!r} in {path}; have {list(df.columns)}")

    rows: list[WeightedCriterion] = []
    for _, row in df.iterrows():
        name = str(row[name_column]).strip()
        if not name or name.lower() == "nan":
            continue
        w = row[weight_column]
        try:
            weight = float(w)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Bad weight for row {name!r}: {w!r}") from e
        body = str(row[body_column]).strip()
        if not body or body.lower() == "nan":
            continue
        rows.append(WeightedCriterion(name=name, weight=weight, body=body))
    return rows


_STANDALONE_POINTS = re.compile(r"^(\d+(?:\.\d+)?)\s*points?\s*$", re.IGNORECASE)
_INLINE_POINTS = re.compile(r"\(([+-]?[\d.]+)\s*points?[^)]*\)", re.IGNORECASE)
# Long line ending with "(N points)" where N > 0 — used only for inline / Part-block heuristics.
_MAIN_POSITIVE_TAIL = re.compile(r"\(([\d.]+)\s*points?\)\s*\.?\s*$", re.IGNORECASE)


def _is_separator_line(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if t in ("---", "—", "___", "***", "..."):
        return True
    return re.match(r"^[\-–_=]{3,}\s*$", t) is not None


def _strip_separator_edges(text: str) -> str:
    lines = text.splitlines()
    while lines and _is_separator_line(lines[0]):
        lines.pop(0)
    while lines and _is_separator_line(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _first_meaningful_line(body: str) -> str:
    for ln in body.splitlines():
        s = ln.strip()
        if not s or _is_separator_line(s):
            continue
        return s
    return ""


def _starts_next_main_section(line: str) -> bool:
    s = line.strip()
    if _STANDALONE_POINTS.match(s):
        return True
    m = _MAIN_POSITIVE_TAIL.search(s)
    if not m:
        return False
    try:
        v = float(m.group(1))
    except ValueError:
        return False
    if v <= 0:
        return False
    before = s[: m.start()].strip()
    return len(before) >= 18


def load_weighted_criteria_txt(path: str | Path) -> list[WeightedCriterion]:
    """
    Best-effort parser for rubrics like gsu-sumprod.txt:
    - A line that is only "N points" starts a section; all following lines belong to that section
      until the next standalone "N points" line. Sub-bullets with "(-3 points)" stay inside the
      same section (one top-level category per point bucket).
    - Lines that are only "---" are treated as separators (ignored for titles, stripped from edges).
    - A non-empty line containing "(N points ..." may still create its own item when not inside
      a standalone section (legacy / mixed-format files).
    Use merge_consecutive_same_weight() if you need several same-weight sections as one category.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    items: list[WeightedCriterion] = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        if not line:
            i += 1
            continue

        m0 = _STANDALONE_POINTS.match(line)
        if m0:
            w = float(m0.group(1))
            i += 1
            buf: list[str] = []
            while i < len(lines):
                nxt = lines[i].strip()
                if _STANDALONE_POINTS.match(nxt):
                    break
                if lines[i].strip():
                    buf.append(lines[i].rstrip())
                i += 1
            body = _strip_separator_edges("\n".join(buf).strip())
            if body:
                title = _first_meaningful_line(body)
                if len(title) > 80:
                    title = title[:77] + "..."
                if not title or _is_placeholder_title(title):
                    title = f"Category ({w:g} pts)"
                items.append(
                    WeightedCriterion(
                        name=title,
                        weight=w,
                        body=body,
                    )
                )
            continue

        m = _INLINE_POINTS.search(line)
        if m and len(line) >= 10:
            w = float(m.group(1))
            pre_lines: list[str] = []
            j = i - 1
            while j >= 0:
                prev = lines[j].rstrip()
                pst = prev.strip()
                if not pst:
                    break
                if _STANDALONE_POINTS.match(pst):
                    break
                if _starts_next_main_section(prev):
                    break
                if _INLINE_POINTS.search(prev) and len(pst) >= 12:
                    break
                pre_lines.insert(0, prev)
                j -= 1
            body_core = line.strip()
            body = "\n".join([*pre_lines, body_core]).strip() if pre_lines else body_core
            title = body.split("\n", 1)[0].strip()
            if len(title) > 80:
                title = title[:77] + "..."
            items.append(WeightedCriterion(name=title or body_core[:80], weight=w, body=body))
            i += 1
            continue

        # "Part a:" / "Part b:" continuations (no points on the line) — attach to previous criterion
        if items and re.match(r"^Part\s+[a-z0-9]+:", line, re.I):
            buf_part = [lines[i].rstrip()]
            i += 1
            while i < len(lines):
                s = lines[i].strip()
                if not s:
                    buf_part.append("")
                    i += 1
                    continue
                if _STANDALONE_POINTS.match(s):
                    break
                if re.match(r"^Part\s+[a-z0-9]+:", s, re.I):
                    buf_part.append(lines[i].rstrip())
                    i += 1
                    continue
                if _starts_next_main_section(lines[i]):
                    break
                if _INLINE_POINTS.search(lines[i]) and len(s) >= 12:
                    break
                buf_part.append(lines[i].rstrip())
                i += 1
            ext = "\n".join(buf_part).strip()
            last = items[-1]
            items[-1] = WeightedCriterion(
                name=last.name,
                weight=last.weight,
                body=(last.body + "\n\n" + ext).strip(),
            )
            continue

        i += 1

    return items


def load_weighted_criteria_path(
    path: str | Path,
    *,
    csv_body_column: str | None = None,
    csv_name_column: str = "Criteria",
    csv_weight_column: str = "Weight",
) -> list[WeightedCriterion]:
    """Load weighted criteria from a .csv (structured) or .txt (heuristic)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_weighted_criteria_csv(
            path,
            name_column=csv_name_column,
            weight_column=csv_weight_column,
            body_column=csv_body_column,
        )
    if suffix in (".txt", ".md"):
        return load_weighted_criteria_txt(path)
    raise ValueError(f"Unsupported rubric file type: {path.suffix} ({path})")
