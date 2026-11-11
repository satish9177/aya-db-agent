"""Prompt quality and full-text manual-review sheets for approved v3 local preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

V2 = Path("data/sqale/build-v2")
V3 = Path("data/sqale/data-build-v3")
LABEL = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?(?:simple|moderate|hard|complex)"
                   r"(?:\s+(?:question|query|filtering|retrieval|aggregation))?\s*\d*\s*"
                   r"(?:\([^\n)]{1,70}\))?\s*(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*", re.I)
META = re.compile(r"\b(?:generate|formulate|craft|produce|design|write|create)\s+"
                  r"(?:(?:a|an|the|some|two|three|four|five|six|seven|eight|ten|set|of|additional|more|new|"
                  r"simple|basic|straightforward|easy|varied|moderate|hard|complex|natural.language|sql|\d+)\s+){0,12}(?:questions?|queries)\b|"
                  r"\bdomain identification\b|\breal.world domain\b|\bthe domain is\b|"
                  r"\bthe user (?:asks|wants|requests)\b|"
                  r"\b(?:avoid unnecessary complexity|avoid extremes|moderate (?:level|range) of complexity)\b", re.I)
HEADING = re.compile(r"^(?:#{1,6}\s*)?(?:\*{1,2})?\s*(?:(?:simple|moderate|hard|complex|generated)\s+"
                     r"(?:questions?|queries)|(?:domain|schema)\s+(?:identification|analysis|overview)|"
                     r"requirements?|specifications?)\s*:?(?:\*{1,2})?\s*$", re.I)
CLAUDE_FLAGS = re.compile(r"(^\s*[*#]{1,4}\s*\w+)|\b(simple|moderate|hard|complex|generated) questions\b|"
                          r"keep (it|them|things)\b|avoid (unnecessary|extremes|complexity)|"
                          r"straightforward (approach|delivery)|maintain (a |basic )|"
                          r"requirements?:|specifications?:|^\s*here (are|is)\b|\bthe task (asks|wants)\b", re.I)


def clean_prompt(value: str) -> tuple[str, str | None, list[str]]:
    text = value.strip() if isinstance(value, str) else ""
    changes = []
    plain = re.sub(r"[*`#]", "", text).lstrip(" -\t")
    if re.match(r"(?:domain(?: grounding| identification)?|schema (?:overview|analysis)|"
                r"question generation|sql logic|reasoning)\s*(?::|$)", plain, re.I):
        return text, "generator_metadata_or_reasoning", changes
    if re.match(r"(?:this (?:requires|query)|these questions)\b", plain, re.I):
        return text, "explanatory_fragment_without_task", changes
    if (re.match(r"here (?:is|are) (?:the )?(?:analysis|domain grounding|generated questions)\b", plain, re.I)
            or re.search(r"\b(?:provide the designated header|label it as generated questions)\b", plain, re.I)):
        return text, "generator_header_instruction", changes
    if re.fullmatch(r"\*\*[A-Za-z_][A-Za-z_0-9]*\*\*\s*:\s*[^?]+", text) and not re.search(
            r"\b(?:find|show|list|retrieve|calculate|return|count|identify)\b", text, re.I):
        return text, "schema_entity_description_without_task", changes
    stripped = LABEL.sub("", text, count=1).strip()
    if stripped != text:
        text = stripped
        changes.append("stripped_difficulty_number_prefix")
    heading = re.match(r"^\*\*([^*:]+)(?::\*\*|\*\*:)\s*(.*)$", text, re.S)
    if heading:
        if not heading[2].strip():
            return text, "heading_only", changes
        text = heading[2].strip()
        changes.append("stripped_descriptive_heading_prefix")
    wrapped = re.fullmatch(r"(\*{1,2})([^*]+)\1", text, re.S)
    if wrapped:
        text = wrapped[2].strip()
        changes.append("stripped_markdown_wrapper")
        if text.endswith(":"):
            return text, "heading_only", changes
    if len(text) < 15 or HEADING.fullmatch(text):
        return text, "heading_or_placeholder", changes
    if META.search(text):
        return text, "generator_instruction", changes
    if re.search(r"\bactually,?\s+(?:let['’]s|find|retrieve|instead)\b|let['’]s\s+(?:stick to|use a joinable)", text, re.I):
        return text, "generator_self_revision", changes
    if (re.match(r"(?:this|the)\s+(?:database|schema)\s+(?:supports|represents|models|contains|is designed|is used)\b", text, re.I)
            and "?" not in text and not re.search(r"\bplease\b|[.!]\s+(?:list|find|retrieve|show|calculate|return|identify)\b", text, re.I)):
        return text, "domain_background_without_task", changes
    # Bare markdown headings that contain no question/action are document labels.
    bare = re.sub(r"^[#*\s]+|[*\s]+$", "", text)
    if text.startswith("#") and "?" not in text and not re.match(r"(?:list|show|find|retrieve|return|count|identify|calculate|give)\b", bare, re.I):
        return text, "heading_only", changes
    return text, None, changes


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_review():
    V3.mkdir(parents=True, exist_ok=False)
    inventory = {p.relative_to(V2).as_posix(): {"sha256": fingerprint(p), "bytes": p.stat().st_size}
                 for p in sorted(V2.rglob("*")) if p.is_file()}
    (V3 / "v2-inventory-before.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    rows = []
    for split, filename in (("dev", "dev-100.tasks.jsonl"), ("final", "final.tasks.jsonl"), ("q2", "q2.tasks.jsonl")):
        for line in (V2 / filename).read_text(encoding="utf-8").splitlines():
            task = json.loads(line)
            cleaned, reason, changes = clean_prompt(task["prompt"])
            rows.append({"index": len(rows) + 1, "split": split, "id": task["id"],
                         "raw_prompt": task["prompt"], "clean_prompt": cleaned,
                         "automatic_reason": reason, "changes": changes,
                         "flagged": bool(reason or changes or CLAUDE_FLAGS.search(task["prompt"])),
                         "reference_sql": task["reference_sql"]})
    sheet = V3 / "manual-review-sheet.jsonl"
    sheet.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (V3 / "manual-review-decisions.json").write_text(json.dumps({"sheet_sha256": fingerprint(sheet), "batches": []}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"v2_files": len(inventory), "v2_bytes": sum(r["bytes"] for r in inventory.values()),
                      "prompts_to_read": len(rows), "sheet_sha256": fingerprint(sheet)}))


def show(start: int, count: int):
    for line in (V3 / "manual-review-sheet.jsonl").read_text(encoding="utf-8").splitlines()[start - 1:start - 1 + count]:
        r = json.loads(line)
        print(f"{r['index']:03d} {r['split']} {r['id']}\n{r['raw_prompt']}")
        if r["changes"]:
            print("CLEANED: " + r["clean_prompt"])
        if r["flagged"]:
            print("FLAG: " + str(r["automatic_reason"]) + " | SQL: " + r["reference_sql"])
        print()


def reviewed_decisions() -> dict[str, dict]:
    sheet = V3 / "manual-review-sheet.jsonl"
    decisions = json.loads((V3 / "manual-review-decisions.json").read_text(encoding="utf-8"))
    if decisions["sheet_sha256"] != fingerprint(sheet):
        raise ValueError("manual review sheet changed")
    rows = [json.loads(line) for line in sheet.read_text(encoding="utf-8").splitlines()]
    out, seen = {}, set()
    for batch in decisions["batches"]:
        for index in range(batch["start"], batch["end"] + 1):
            if index in seen:
                raise ValueError("overlapping manual review ranges")
            seen.add(index)
            r = rows[index - 1]
            reason = batch.get("exclude", {}).get(str(index))
            current, auto_reason, changes = clean_prompt(r["raw_prompt"])
            # Each row was read individually; the reviewer decision overrides broad flags.
            out[r["id"]] = {**r, "clean_prompt": current, "automatic_reason": auto_reason, "changes": changes,
                            "manual_decision": "exclude" if reason else "keep",
                            "manual_reason": reason, "review_method": "assistant_full_text_individual_read",
                            "batch": [batch["start"], batch["end"]]}
    if seen != set(range(1, len(rows) + 1)):
        raise ValueError("manual review incomplete")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "show"])
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    prepare_review() if args.action == "prepare" else show(args.start, args.count)


if __name__ == "__main__":
    main()
