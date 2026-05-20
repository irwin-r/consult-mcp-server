You have {n} expert reviews of a code artefact (PR diff, file, or codebase). Synthesise them as a single PR/code review under this rubric:

# Blockers
Issues that must be fixed before merge. For each finding: cite source slugs in [brackets], state severity (security / correctness / data-loss / performance), name the file and line range when known, and the specific change required. Group by file when multiple blockers target the same file.

# Suggestions
Improvements that are not blockers but worth considering. Group by category (correctness, performance, maintainability, style, tests). Cite slugs. Be concrete — "rename X to Y" beats "consider renaming".

# Praise
What the change does well. Surface this — it's calibration signal for the author and a marker of what to preserve in future changes.

# Open Questions
Genuine uncertainty across the panel where the panellists could not converge. State what would resolve each one (a clarification from the author, a benchmark, a design discussion, an existing convention to consult).

# Verdict
One of: SHIP / CHANGES_REQUESTED / DISCUSS. Justify in one sentence.

Be specific. Quote line numbers, function names, and variable names verbatim from the panel where it helps. If panellists disagree on whether something is a blocker or a suggestion, surface that disagreement explicitly. Down-weight responses tagged TRUNCATED or with confidence < 0.4.
