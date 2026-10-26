"""
mitigations/input_filter.py

Applied to the assembled context BEFORE it reaches the model (FR4).

v2 hardening over the original 8-phrase filter:
  * Text is normalised before matching (Unicode NFKC, zero-width and
    direction-control characters removed, whitespace collapsed), so simple
    obfuscation such as "ig​nore" or full-width letters no longer evades it.
  * Pattern families are generic prompt-injection tropes (instruction override,
    role/authority claims, chat-template tokens, secret-disclosure requests,
    verdict/label steering, privileged-action requests), written from public
    injection taxonomies rather than tuned to this repo's payload corpus, so
    results are not inflated by overfitting the filter to the test set.

It never deletes content (that would hide real attacks from reviewers); it
wraps matching lines in explicit untrusted-data delimiters. The pairing with
prompt hardening / spotlighting is what tells the model how to treat them.
"""

import re
import unicodedata

_PATTERN_FAMILIES = {
    "instruction_override": [
        r"ignore (all |any |the )?(above|previous|prior|earlier)",
        r"disregard (all |any |the )?(above|previous|prior|earlier|instructions)",
        r"forget (all |everything |your )?(above|previous|prior|instructions)",
        r"(new|updated|revised|real) instructions?\s*[:\-]",
        r"override (token|instructions?|policy|rules?)",
    ],
    "authority_or_role_claim": [
        r"system (note|message|prompt|override)",
        r"assistant[- ]instructions?",
        r"urgent (assistant|ai|bot) instruction",
        r"you are now\b",
        r"(act|behave|respond) as (if you (are|were)|an?|the)\b",
        r"debug mode",
        r"(maintainer|admin(istrator)?|security team|owner) (has )?(approved|authori[sz]ed|instructs?|requires?)",
    ],
    "chat_template_tokens": [
        r"<\|?(im_start|im_end|system|assistant|endoftext)\|?>",
        r"\[/?(INST|SYS)\]",
        r"^\s*#{1,4}\s*(system|assistant)\b",
        r"^\s*(system|assistant)\s*:",
    ],
    "secret_disclosure_request": [
        r"(print|reveal|show|output|include|leak|send|echo|dump) .{0,40}(secret|token|api[ _-]?key|credential|password|environment variable|env\b)",
        r"diagnostic dump",
    ],
    "verdict_or_label_steering": [
        r"always (approve|merge|accept)",
        r"(approve|merge|accept) (this|the|it)\b.{0,30}(pr|pull request|change|without)",
        r"(mark|label|classify|set|rate) (this|it|the pr)? ?(as )?.{0,15}(p3|low priority|chore|not (a )?security)",
        r"(reject|deny|block|request changes).{0,40}(regardless|no matter|always)",
        r"without (review|testing|further (review|checks?))",
    ],
    "privileged_action_request": [
        r"deploy(ing)? .{0,30}(prod|production)",
        r"merge (pr|pull request) ?#?\d+",
        r"grant .{0,30}(admin|write|access|permission)",
        r"(run|execute|invoke|call) .{0,30}(tool|command|script|shell)",
    ],
    "concealment": [
        r"(do not|don't|never) (mention|tell|reveal|disclose|report) .{0,40}",
        r"keep (this|it) (secret|hidden|between us)",
    ],
}

_COMPILED = [re.compile(p, re.IGNORECASE) for fam in _PATTERN_FAMILIES.values() for p in fam]

_INVISIBLE = dict.fromkeys(
    [0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E, 0x2066, 0x2067, 0x2068, 0x2069]
)


def _normalise(line: str) -> str:
    """Match against a canonical form; the ORIGINAL line is what gets output."""
    line = unicodedata.normalize("NFKC", line).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", line)


def is_suspicious(line: str) -> bool:
    norm = _normalise(line)
    return any(p.search(norm) for p in _COMPILED)


def input_filter(text: str) -> str:
    """Wraps any line matching a suspicious pattern in explicit
    untrusted-data delimiters. Content is preserved verbatim."""
    out_lines = []
    for line in text.split("\n"):
        if is_suspicious(line):
            out_lines.append(f"<UNTRUSTED_DATA_FLAGGED>{line}</UNTRUSTED_DATA_FLAGGED>")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)
