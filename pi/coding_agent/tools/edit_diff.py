"""edit diff logic: exact-text replacement with uniqueness and overlap checks.

Python port of ``packages/coding-agent/src/core/tools/edit-diff.ts``.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import List, Tuple


def normalize_to_lf(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def restore_line_endings(content: str, ending: str) -> str:
    if ending == "\n":
        return content
    return content.replace("\n", ending)


@dataclass
class Edit:
    old_text: str
    new_text: str


@dataclass
class MatchedEdit:
    edit_index: int
    match_index: int
    match_length: int
    new_text: str


_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_SPACES = "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
_QUOTES = "\u2018\u2019\u201a\u201b"
_DQUOTES = "\u201c\u201d\u201e\u201f"


def normalize_for_fuzzy_match(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    normalized = "\n".join(lines)
    for ch in _QUOTES:
        normalized = normalized.replace(ch, "'")
    for ch in _DQUOTES:
        normalized = normalized.replace(ch, '"')
    for ch in _DASHES:
        normalized = normalized.replace(ch, "-")
    for ch in _SPACES:
        normalized = normalized.replace(ch, " ")
    return normalized


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False, index=-1, match_length=0, used_fuzzy_match=False, content_for_replacement=content
        )
    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


def count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    if not fuzzy_old_text:
        return 0
    return fuzzy_content.count(fuzzy_old_text)


def _not_found_error(path: str, edit_index: int, total: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total: int, occurrences: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. "
            "The text must be unique. Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total: int) -> ValueError:
    if total == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _no_change_error(path: str, total: int) -> ValueError:
    if total == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(f"No changes made to {path}. The replacements produced identical content.")


def _split_lines_with_endings(content: str) -> List[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


def _line_spans(content: str) -> List[Tuple[int, int]]:
    offset = 0
    spans: List[Tuple[int, int]] = []
    for line in _split_lines_with_endings(content):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _replacement_line_range(spans: List[Tuple[int, int]], match_index: int, match_length: int) -> Tuple[int, int]:
    replacement_end = match_index + match_length
    start_line = -1
    for i, (start, end) in enumerate(spans):
        if start <= match_index < end:
            start_line = i
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(spans) and spans[end_line][1] < replacement_end:
        end_line += 1
    if end_line >= len(spans):
        raise ValueError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_replacements(content: str, matched: List[MatchedEdit], offset: int = 0) -> str:
    result = content
    for replacement in reversed(matched):
        index = replacement.match_index - offset
        result = (
            result[:index] + replacement.new_text + result[index + replacement.match_length:]
        )
    return result


def _apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, matched: List[MatchedEdit]
) -> str:
    original_lines = _split_lines_with_endings(original_content)
    spans = _line_spans(base_content)
    if len(original_lines) != len(spans):
        raise ValueError("Cannot preserve unchanged lines because the base content has a different line count.")

    groups: List[dict] = []
    for replacement in sorted(matched, key=lambda m: m.match_index):
        start_line, end_line = _replacement_line_range(spans, replacement.match_index, replacement.match_length)
        current = groups[-1] if groups else None
        if current and start_line < current["end_line"]:
            current["end_line"] = max(current["end_line"], end_line)
            current["replacements"].append(replacement)
        else:
            groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index: group["start_line"]])
        group_start = spans[group["start_line"]][0]
        group_end = spans[group["end_line"] - 1][1]
        result += _apply_replacements(base_content[group_start:group_end], group["replacements"], group_start)
        original_line_index = group["end_line"]
    result += "".join(original_lines[original_line_index:])
    return result


def apply_edits_to_normalized_content(
    normalized_content: str, edits: List[Edit], path: str
) -> Tuple[str, str]:
    """Apply exact-text replacements; returns (base_content, new_content)."""
    normalized_edits = [Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text)) for e in edits]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, e.old_text) for e in normalized_edits]
    used_fuzzy_match = any(m.used_fuzzy_match for m in initial_matches)
    base_for_replacement = normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content

    matched: List[MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        match = fuzzy_find_text(base_for_replacement, edit.old_text)
        if not match.found:
            raise _not_found_error(path, i, len(normalized_edits))
        occurrences = count_occurrences(base_for_replacement, edit.old_text)
        if occurrences > 1:
            raise _duplicate_error(path, i, len(normalized_edits), occurrences)
        matched.append(
            MatchedEdit(
                edit_index=i, match_index=match.index, match_length=match.match_length, new_text=edit.new_text
            )
        )

    matched.sort(key=lambda m: m.match_index)
    for i in range(1, len(matched)):
        previous, current = matched[i - 1], matched[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    if used_fuzzy_match:
        new_content = _apply_replacements_preserving_unchanged_lines(
            normalized_content, base_for_replacement, matched
        )
    else:
        new_content = _apply_replacements(base_for_replacement, matched)

    if base_content == new_content:
        raise _no_change_error(path, len(normalized_edits))
    return base_content, new_content


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(diff)


def generate_diff_string(old_content: str, new_content: str, context_lines: int = 2) -> Tuple[str, int]:
    """Render a compact diff with the first changed line (1-based, new file)."""

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    hunks: List[str] = []
    first_changed_line: int = 1

    for group in matcher.get_grouped_opcodes(context_lines):
        old_start = max(group[0][1] + 1 - 0, 1)
        old_end = group[-1][2]
        new_start = group[0][3] + 1
        if len(hunks) == 0:
            first_changed_line = new_start
        hunk_lines = [f"@@ -{old_start},{old_end - old_start + 1} +{new_start},{group[-1][4] - new_start + 1} @@"]
        new_line = new_start
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                new_line += j2 - j1
                continue
            for line in old_lines[i1:i2]:
                hunk_lines.append(f"-{line}")
            for line in new_lines[j1:j2]:
                hunk_lines.append(f"+{line}")
            new_line += j2 - j1
        hunks.append("\n".join(hunk_lines))

    diff = "\n".join(hunks)
    return diff, first_changed_line
