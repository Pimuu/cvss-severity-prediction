"""CVSS v3.1 base-score calculator and qualitative severity mapping.

Implements the base-score equations from the FIRST CVSS v3.1 specification
(Section 7). Used by the data generator so that every synthetic record's label
is derived from a *valid* CVSS vector rather than assigned by hand.
"""
import math

# Metric weights (CVSS v3.1 specification, Section 7.4) ---------------------
AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}       # Attack Vector
AC = {"L": 0.77, "H": 0.44}                              # Attack Complexity
PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}         # Privileges Required, Scope Unchanged
PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}           # Privileges Required, Scope Changed
UI = {"N": 0.85, "R": 0.62}                              # User Interaction
CIA = {"H": 0.56, "L": 0.22, "N": 0.0}                   # Confidentiality / Integrity / Availability


def roundup(x: float) -> float:
    """CVSS v3.1 'Roundup': smallest 1-decimal number >= x (float-safe, Appendix A)."""
    i = int(round(x * 100000))
    if i % 10000 == 0:
        return i / 100000.0
    return (math.floor(i / 10000) + 1) / 10.0


def base_score(m: dict) -> float:
    """Compute the CVSS v3.1 base score for a metric dict.

    `m` must contain keys AV, AC, PR, UI, S, C, I, A with single-letter values,
    e.g. {"AV": "N", "AC": "L", "PR": "N", "UI": "N", "S": "U",
          "C": "H", "I": "H", "A": "H"}  -> 9.8
    """
    changed = m["S"] == "C"
    iss = 1 - (1 - CIA[m["C"]]) * (1 - CIA[m["I"]]) * (1 - CIA[m["A"]])
    if changed:
        impact = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15
    else:
        impact = 6.42 * iss
    pr = (PR_CHANGED if changed else PR_UNCHANGED)[m["PR"]]
    exploitability = 8.22 * AV[m["AV"]] * AC[m["AC"]] * pr * UI[m["UI"]]
    if impact <= 0:
        return 0.0
    if changed:
        return roundup(min(1.08 * (impact + exploitability), 10))
    return roundup(min(impact + exploitability, 10))


def severity(score: float) -> str:
    """Map a base score to the CVSS v3.x qualitative rating."""
    if score == 0.0:
        return "None"
    if score < 4.0:
        return "Low"
    if score < 7.0:
        return "Medium"
    if score < 9.0:
        return "High"
    return "Critical"


def vector_string(m: dict) -> str:
    """Render a metric dict as a standard CVSS:3.1 vector string."""
    keys = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
    return "CVSS:3.1/" + "/".join(f"{k}:{m[k]}" for k in keys)
