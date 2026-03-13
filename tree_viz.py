"""Render an RKTRoot skill tree as a visual ASCII tree."""

from type import RKTRoot, RKTRubricLine, RKTBasicRule, RKTSimpleRule


def _rule_text(rule) -> str:
    """Get display text for a rule from description or rule field."""
    text = (getattr(rule, "description", None) or getattr(rule, "rule", None) or "").strip()
    return text[:67] + "..." if len(text) > 70 else text


def _format_rule(rule, prefix: str, is_last: bool) -> list[str]:
    """Recursively format a Rule (basic or simple) into tree lines."""
    lines = []
    branch = "└── " if is_last else "├── "
    if rule.type == "simple rule":
        text = _rule_text(rule)
        label = text or "[simple rule]"
        lines.append(prefix + branch + label)
    else:
        # Basic rule: show description (or fallback), then children
        desc = _rule_text(rule)
        label = desc or f"(group: {len(getattr(rule, 'children', []) or [])} items)"
        lines.append(prefix + branch + label)
        child_prefix = prefix + ("    " if is_last else "│   ")
        children = getattr(rule, "children", []) or []
        for i, child in enumerate(children):
            lines.extend(_format_rule(child, child_prefix, i == len(children) - 1))
    return lines


def _format_rubric_line(line: RKTRubricLine, prefix: str, is_last: bool) -> list[str]:
    """Format a rubric line (top-level category) and its rule children."""
    lines = []
    branch = "└── " if is_last else "├── "
    desc = (line.description or "").strip()
    if len(desc) > 70:
        desc = desc[:67] + "..."
    lines.append(prefix + branch + f"📁 {desc}")
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = getattr(line, "children", []) or []
    for i, child in enumerate(children):
        lines.extend(_format_rule(child, child_prefix, i == len(children) - 1))
    return lines


def render_skill_tree(root: RKTRoot) -> str:
    """Turn an RKTRoot into a readable ASCII tree string."""
    lines = ["Skill tree", "=========="]
    rows = getattr(root, "rows", []) or []
    for i, row in enumerate(rows):
        lines.extend(_format_rubric_line(row, "", is_last=(i == len(rows) - 1)))
    return "\n".join(lines)
