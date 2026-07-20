import json


def extract_plain_text(raw: str | None) -> str | None:
    """Converts a Slate/Plate rich-text jsonb column (double-JSON-encoded: the
    column value is a JSON string containing a JSON-encoded node array) into
    plain text. Used for activity/comment tables that only store rich content
    with no precomputed plain-text column (unlike Announcement.contentText).

    Node shapes seen in this data: {"text": "..."} leaves, {"type": "mention",
    "value": "Name", "children": [...]} mentions (use value, not the empty
    text leaf inside), and block/inline wrappers with a "children" list.
    """
    if not raw:
        return None

    try:
        nodes = json.loads(raw)
        if isinstance(nodes, str):
            nodes = json.loads(nodes)
    except (json.JSONDecodeError, TypeError):
        return None

    parts: list[str] = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("type") == "mention":
            value = node.get("value")
            if value:
                parts.append(f"@{value}")
            return  # skip its children - they're just an empty placeholder text leaf
        if "text" in node:
            if node["text"]:
                parts.append(node["text"])
            return
        if "children" in node:
            walk(node["children"])

    walk(nodes)
    text = " ".join(parts).strip()
    return text or None
