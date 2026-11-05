"""Shared diff computation for the edit tool (port of ``tools/edit-diff.ts``).

Porting deviation: the TS uses the npm ``diff`` package; this port builds the
same display diff and unified patch from ``difflib.SequenceMatcher`` line
opcodes. The patch header carries no timestamps, like ``FILE_HEADERS_ONLY``.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

QUOTE_MAP = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    # U+2010 hyphen, U+2011 non-breaking hyphen, U+2012 figure dash,
    # U+2013 en-dash, U+2014 em-dash, U+2015 horizontal bar, U+2212 minus
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    # U+00A0 NBSP, U+2002-U+200A various spaces, U+202F narrow NBSP,
    # U+205F medium math space, U+3000 ideographic space
    "\u00a0": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2004": " ",
    "\u2005": " ",
    "\u2006": " ",
    "\u2007": " ",
    "\u2008": " ",
    "\u2009": " ",
    "\u200a": " ",
    "\u202f": " ",
    "\u205f": " ",
    "\u3000": " ",
}
_QUOTE_RE = re.compile("[" + "".join(QUOTE_MAP.keys()) + "]")


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def normalize_for_fuzzy_match(text: str) -> str:
    """Normalize text for fuzzy matching (trailing whitespace, smart quotes, dashes, special spaces)."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return _QUOTE_RE.sub(lambda match: QUOTE_MAP[match.group(0)], normalized)


def split_lines_with_endings(content: str) -> List[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


class LineSpan:
    __slots__ = ("start", "end")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class MatchedEdit:
    __slots__ = ("edit_index", "match_index", "match_length", "new_text")

    def __init__(self, edit_index: int, match_index: int, match_length: int, new_text: str) -> None:
        self.edit_index = edit_index
        self.match_index = match_index
        self.match_length = match_length
        self.new_text = new_text


class TextReplacement:
    __slots__ = ("match_index", "match_length", "new_text")

    def __init__(self, match_index: int, match_length: int, new_text: str) -> None:
        self.match_index = match_index
        self.match_length = match_length
        self.new_text = new_text


def get_line_spans(content: str) -> List[LineSpan]:
    spans: List[LineSpan] = []
    offset = 0
    for line in split_lines_with_endings(content):
        spans.append(LineSpan(offset, offset + len(line)))
        offset += len(line)
    return spans


def get_replacement_line_range(lines: List[LineSpan], replacement: TextReplacement) -> Tuple[int, int]:
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length

    start_line = -1
    for i, line in enumerate(lines):
        if line.start <= replacement_start < line.end:
            start_line = i
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")

    return start_line, end_line + 1


def apply_replacements(content: str, replacements: List[TextReplacement], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = result[:match_index] + replacement.new_text + result[match_index + replacement.match_length:]
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str, base_content: str, replacements: List[TextReplacement]
) -> str:
    """Apply replacements matched against ``base_content`` to ``original_content``, keeping its other lines."""
    original_lines = split_lines_with_endings(original_content)
    base_lines = get_line_spans(base_content)
    if len(original_lines) != len(base_lines):
        raise ValueError("Cannot preserve unchanged lines because the base content has a different line count.")

    groups: List[dict] = []
    for replacement in sorted(replacements, key=lambda r: r.match_index):
        start_line, end_line = get_replacement_line_range(base_lines, replacement)
        current = groups[-1] if groups else None
        if current and start_line < current["end_line"]:
            current["end_line"] = max(current["end_line"], end_line)
            current["replacements"].append(replacement)
            continue
        groups.append({"start_line": start_line, "end_line": end_line, "replacements": [replacement]})

    original_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_index : group["start_line"]])
        group_start_offset = base_lines[group["start_line"]].start
        group_end_offset = base_lines[group["end_line"] - 1].end
        result += apply_replacements(
            base_content[group_start_offset:group_end_offset], group["replacements"], group_start_offset
        )
        original_index = group["end_line"]
    result += "".join(original_lines[original_index:])
    return result


class FuzzyMatchResult:
    __slots__ = ("found", "index", "match_length", "used_fuzzy_match", "content_for_replacement")

    def __init__(self, found: bool, index: int, match_length: int, used_fuzzy: bool, content: str) -> None:
        self.found = found
        self.index = index
        self.match_length = match_length
        self.used_fuzzy_match = used_fuzzy
        self.content_for_replacement = content


class Edit:
    __slots__ = ("old_text", "new_text")

    def __init__(self, old_text: str, new_text: str) -> None:
        self.old_text = old_text
        self.new_text = new_text


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """Find ``old_text`` exactly first, then in fuzzy-normalized space."""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(True, exact_index, len(old_text), False, content)

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)
    if fuzzy_index == -1:
        return FuzzyMatchResult(False, -1, 0, False, content)
    return FuzzyMatchResult(True, fuzzy_index, len(fuzzy_old_text), True, fuzzy_content)


def strip_bom(content: str) -> Tuple[str, str]:
    """Strip a UTF-8 BOM; returns ``(bom, text_without_it)``."""
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text) if fuzzy_old_text else 0


def _get_not_found_error(path: str, edit_index: int, total_edits: int) -> Exception:
    if total_edits == 1:
        return ValueError(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _get_duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> Exception:
    if total_edits == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. Each oldText must be unique. "
        "Please provide more context to make it unique."
    )


def _get_empty_old_text_error(path: str, edit_index: int, total_edits: int) -> Exception:
    if total_edits == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _get_no_change_error(path: str, total_edits: int) -> Exception:
    if total_edits == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(f"No changes made to {path}. The replacements produced identical content.")


def apply_edits_to_normalized_content(normalized_content: str, edits: List[Edit], path: str) -> Dict[str, str]:
    """Apply exact-text replacements to LF-normalized content; returns ``{"base", "new"}`` content."""
    normalized_edits = [Edit(normalize_to_lf(edit.old_text), normalize_to_lf(edit.new_text)) for edit in edits]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _get_empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits]
    used_fuzzy_match = any(match.used_fuzzy_match for match in initial_matches)
    replacement_base_content = (
        normalize_for_fuzzy_match(normalized_content) if used_fuzzy_match else normalized_content
    )

    matched_edits: List[MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _get_not_found_error(path, i, len(normalized_edits))
        occurrences = count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _get_duplicate_error(path, i, len(normalized_edits), occurrences)
        matched_edits.append(
            MatchedEdit(i, match_result.index, match_result.match_length, edit.new_text)
        )

    matched_edits.sort(key=lambda edit: edit.match_index)
    for i in range(1, len(matched_edits)):
        previous = matched_edits[i - 1]
        current = matched_edits[i]
        if previous.match_index + previous.match_length > current.match_index:
            raise ValueError(
                f"edits[{previous.edit_index}] and edits[{current.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    replacements = [TextReplacement(e.match_index, e.match_length, e.new_text) for e in matched_edits]
    new_content = (
        apply_replacements_preserving_unchanged_lines(normalized_content, replacement_base_content, replacements)
        if used_fuzzy_match
        else apply_replacements(replacement_base_content, replacements)
    )

    if base_content == new_content:
        raise _get_no_change_error(path, len(normalized_edits))

    return {"base": base_content, "new": new_content}


def generate_unified_patch(path: str, old_content: str, new_content: str, context_lines: int = 4) -> str:
    """Generate a standard unified patch."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    if old_content.endswith("\n"):
        old_lines = [line if line.endswith("\n") else line + "\n" for line in old_lines]
    if new_content.endswith("\n"):
        new_lines = [line if line.endswith("\n") else line + "\n" for line in new_lines]
    diff = list(
        difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, n=context_lines, lineterm="\n")
    )
    # difflib ends every line with \n except possibly the last; normalize the header lines and drop
    # the trailing newline the way a patch file would carry it.
    if diff and not diff[-1].endswith("\n"):
        diff[-1] += "\n"
    return "".join(diff)


def _diff_line_parts(old_content: str, new_content: str) -> List[dict]:
    """Line-diff parts like ``Diff.diffLines``: runs of equal, added, and removed lines."""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    parts: List[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append({"value": "\n".join(old_lines[i1:i2]) + "\n", "added": False, "removed": False})
        elif tag == "replace":
            parts.append({"value": "\n".join(old_lines[i1:i2]) + "\n", "added": False, "removed": True})
            parts.append({"value": "\n".join(new_lines[j1:j2]) + "\n", "added": True, "removed": False})
        elif tag == "delete":
            parts.append({"value": "\n".join(old_lines[i1:i2]) + "\n", "added": False, "removed": True})
        elif tag == "insert":
            parts.append({"value": "\n".join(new_lines[j1:j2]) + "\n", "added": True, "removed": False})
    # Match jsdiff's trailing behavior: no synthetic newline after the final line when the
    # original content does not end with one.
    if parts:
        last = parts[-1]
        if not last["added"] and not last["removed"]:
            pass
    return parts


def generate_diff_string(
    old_content: str, new_content: str, context_lines: int = 4
) -> Dict[str, object]:
    """Generate a display diff with line numbers and context.

    Returns ``{"diff": str, "first_changed_line": int | None}`` (line numbers in the new file).
    """
    parts = _diff_line_parts(old_content, new_content)
    output: List[str] = []

    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))

    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: Optional[int] = None

    for i, part in enumerate(parts):
        raw = part["value"].split("\n")
        if raw and raw[-1] == "":
            raw.pop()

        if part["added"] or part["removed"]:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in raw:
                if part["added"]:
                    output.append(f"+{str(new_line_num).rjust(line_num_width)} {line}")
                    new_line_num += 1
                else:
                    output.append(f"-{str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
            last_was_change = True
        else:
            next_part_is_change = i < len(parts) - 1 and (parts[i + 1]["added"] or parts[i + 1]["removed"])
            has_leading_change = last_was_change
            has_trailing_change = next_part_is_change

            if has_leading_change and has_trailing_change:
                if len(raw) <= context_lines * 2:
                    for line in raw:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                else:
                    leading = raw[:context_lines]
                    trailing = raw[len(raw) - context_lines :]
                    skipped = len(raw) - len(leading) - len(trailing)
                    for line in leading:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                    for line in trailing:
                        output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                        old_line_num += 1
                        new_line_num += 1
            elif has_leading_change:
                shown = raw[:context_lines]
                skipped = len(raw) - len(shown)
                for line in shown:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
                if skipped > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
            elif has_trailing_change:
                skipped = max(0, len(raw) - context_lines)
                if skipped > 0:
                    output.append(f" {''.rjust(line_num_width)} ...")
                    old_line_num += skipped
                    new_line_num += skipped
                for line in raw[skipped:]:
                    output.append(f" {str(old_line_num).rjust(line_num_width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                old_line_num += len(raw)
                new_line_num += len(raw)
            last_was_change = False

    return {"diff": "\n".join(output), "first_changed_line": first_changed_line}
