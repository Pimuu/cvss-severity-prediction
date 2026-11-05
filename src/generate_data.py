"""Synthetic CVE-style vulnerability description generator.

Design (see README / paper Section 4.1 for the full rationale)
---------------------------------------------------------------
Real NVD descriptions are short, formulaic English sentences
("<weakness> in <component> in <product> <versions> allows <actor> to <impact>
via <vector>.") whose severity is *not* stated in the text: it comes from an
analyst-assigned CVSS vector. We mimic that causal structure:

1. Sample a vulnerability class (a CWE-like weakness family) from a prior.
2. Sample a latent CVSS v3.1 vector (AV, AC, PR, UI, S, C, I, A) from
   class-conditional distributions (e.g. XSS -> UI:R, S:C, C:L/I:L).
3. Compute the official CVSS v3.1 base score and map it to Low/Medium/High/
   Critical. **The label is derived from the vector, never chosen directly.**
4. Render a description from the class and vector. Each metric is only
   *partially observable*: it is verbalised with a probability < 1
   (e.g. "unauthenticated remote attackers" vs. the uninformative "an attacker"),
   and ~10% of records use a vague template that reveals almost nothing.
5. With probability LABEL_NOISE one hidden metric is re-sampled *after*
   rendering, emulating analyst disagreement between text and score.

Steps 4-5 make the task realistically imperfect: the text constrains but does
not determine the label. Product/vendor names are random pseudo-words so they
carry no signal and do not refer to real software.
"""
import numpy as np
import pandas as pd

from . import config
from .cvss import base_score, severity, vector_string

# ---------------------------------------------------------------------------
# 1. Vulnerability classes: prior weight, CVSS metric distributions, phrases.
#    Metric distributions are {value: probability}; omitted metrics use DEFAULT.
# ---------------------------------------------------------------------------
DEFAULT = {
    "AV": {"N": 0.75, "A": 0.05, "L": 0.18, "P": 0.02},
    "AC": {"L": 0.85, "H": 0.15},
    "PR": {"N": 0.55, "L": 0.35, "H": 0.10},
    "UI": {"N": 0.80, "R": 0.20},
    "S": {"U": 0.90, "C": 0.10},
    "C": {"H": 0.50, "L": 0.25, "N": 0.25},
    "I": {"H": 0.50, "L": 0.25, "N": 0.25},
    "A": {"H": 0.50, "L": 0.15, "N": 0.35},
}

CLASSES = {
    "os_command_injection": dict(
        weight=6, cwe="CWE-78",
        names=["OS command injection", "command injection", "shell metacharacter injection"],
        bug=["does not properly sanitize user-supplied input before passing it to a system shell",
             "passes unsanitized parameters to a shell command"],
        components=["the diagnostic ping utility", "the web management interface", "the backup scheduler",
                    "the firmware update handler", "the network configuration page"],
        metrics={"AV": {"N": 0.85, "A": 0.10, "L": 0.05}, "PR": {"N": 0.45, "L": 0.35, "H": 0.20},
                 "C": {"H": 0.95, "L": 0.05}, "I": {"H": 0.95, "L": 0.05}, "A": {"H": 0.9, "L": 0.1}},
        impact=["execute arbitrary OS commands", "execute arbitrary commands", "achieve remote code execution"]),
    "code_injection": dict(
        weight=4, cwe="CWE-94",
        names=["code injection", "template injection", "expression language injection", "eval injection"],
        bug=["evaluates user-controlled template expressions", "passes untrusted input to an eval call"],
        components=["the template engine", "the plugin loader", "the report generator", "the scripting console"],
        metrics={"AV": {"N": 0.9, "L": 0.1}, "PR": {"N": 0.5, "L": 0.3, "H": 0.2},
                 "C": {"H": 0.95, "L": 0.05}, "I": {"H": 0.95, "L": 0.05}, "A": {"H": 0.9, "N": 0.1}},
        impact=["execute arbitrary code", "execute arbitrary PHP code", "execute arbitrary Java code"]),
    "deserialization": dict(
        weight=3, cwe="CWE-502",
        names=["insecure deserialization", "deserialization of untrusted data", "unsafe object deserialization"],
        bug=["deserializes untrusted objects without validation", "accepts serialized Java objects from the network"],
        components=["the RPC endpoint", "the session handler", "the message queue listener", "the cluster sync service"],
        metrics={"AV": {"N": 0.9, "A": 0.05, "L": 0.05}, "PR": {"N": 0.6, "L": 0.3, "H": 0.1},
                 "AC": {"L": 0.75, "H": 0.25},
                 "C": {"H": 0.95, "L": 0.05}, "I": {"H": 0.95, "L": 0.05}, "A": {"H": 0.95, "L": 0.05}},
        impact=["execute arbitrary code", "achieve remote code execution", "run arbitrary commands"]),
    "memory_corruption": dict(
        weight=13, cwe="CWE-787",
        names=["heap-based buffer overflow", "stack-based buffer overflow", "out-of-bounds write",
               "use-after-free", "integer overflow", "double free", "out-of-bounds read"],
        bug=["fails to validate a length field", "miscalculates a buffer size", "frees an object that is still referenced",
             "does not check array bounds"],
        components=["the image decoder", "the font parser", "the PDF renderer", "the packet parser",
                    "the video codec", "the archive extraction library", "the kernel driver", "the XML parser"],
        metrics={"AV": {"N": 0.45, "L": 0.45, "A": 0.07, "P": 0.03}, "UI": {"N": 0.55, "R": 0.45},
                 "PR": {"N": 0.75, "L": 0.2, "H": 0.05},
                 "C": {"H": 0.6, "L": 0.15, "N": 0.25}, "I": {"H": 0.6, "L": 0.1, "N": 0.3},
                 "A": {"H": 0.9, "L": 0.1}},
        impact=None),  # generic CIA-based impact
    "sql_injection": dict(
        weight=10, cwe="CWE-89",
        names=["SQL injection", "blind SQL injection", "time-based SQL injection", "SQL injection vulnerability"],
        bug=["builds SQL queries by concatenating user input", "does not neutralize special elements in an SQL command"],
        components=["the search function", "the login form", "the reporting module", "the order history page",
                    "the REST API", "the admin dashboard"],
        metrics={"AV": {"N": 0.97, "A": 0.03}, "PR": {"N": 0.5, "L": 0.35, "H": 0.15},
                 "C": {"H": 0.9, "L": 0.1}, "I": {"H": 0.7, "L": 0.2, "N": 0.1},
                 "A": {"H": 0.6, "L": 0.1, "N": 0.3}},
        impact=["execute arbitrary SQL commands", "read and modify database contents",
                "extract sensitive data from the database"]),
    "xss": dict(
        weight=16, cwe="CWE-79",
        names=["cross-site scripting (XSS)", "stored cross-site scripting", "reflected cross-site scripting (XSS)",
               "DOM-based cross-site scripting"],
        bug=["does not properly escape user-supplied input in generated HTML", "renders unsanitized comment fields"],
        components=["the comment section", "the profile page", "the search results page", "the admin dashboard",
                    "the file manager", "the error page"],
        metrics={"AV": {"N": 1.0}, "PR": {"N": 0.55, "L": 0.35, "H": 0.10}, "UI": {"R": 0.97, "N": 0.03},
                 "S": {"C": 0.92, "U": 0.08}, "C": {"L": 0.9, "H": 0.05, "N": 0.05},
                 "I": {"L": 0.9, "H": 0.05, "N": 0.05}, "A": {"N": 0.97, "L": 0.03}},
        impact=["inject arbitrary web script or HTML", "execute arbitrary JavaScript in a victim's browser",
                "steal session cookies"]),
    "csrf": dict(
        weight=4, cwe="CWE-352",
        names=["cross-site request forgery (CSRF)", "CSRF", "missing CSRF token validation"],
        bug=["does not verify anti-CSRF tokens", "accepts state-changing GET requests"],
        components=["the settings page", "the user management page", "the password change form", "the plugin settings"],
        metrics={"AV": {"N": 1.0}, "PR": {"N": 0.95, "L": 0.05}, "UI": {"R": 1.0},
                 "C": {"H": 0.35, "L": 0.25, "N": 0.4}, "I": {"H": 0.5, "L": 0.5},
                 "A": {"H": 0.3, "L": 0.1, "N": 0.6}},
        impact=["hijack the authentication of administrators", "perform unauthorized actions on behalf of the victim",
                "change account settings"]),
    "auth_bypass": dict(
        weight=6, cwe="CWE-287",
        names=["authentication bypass", "improper authentication", "missing authentication for critical function",
               "broken authentication"],
        bug=["does not require authentication for a critical function", "improperly validates session tokens",
             "accepts a forged JWT signature"],
        components=["the administrative API", "the login handler", "the remote management service", "the SSO module"],
        metrics={"AV": {"N": 0.9, "A": 0.1}, "PR": {"N": 1.0}, "AC": {"L": 0.85, "H": 0.15},
                 "C": {"H": 0.8, "L": 0.2}, "I": {"H": 0.75, "L": 0.15, "N": 0.1},
                 "A": {"H": 0.6, "L": 0.1, "N": 0.3}},
        impact=["bypass authentication and gain administrative access", "gain unauthorized access",
                "access the administrative interface without credentials"]),
    "hardcoded_credentials": dict(
        weight=2, cwe="CWE-798",
        names=["hard-coded credentials", "a hard-coded password", "default credentials", "a hard-coded cryptographic key"],
        bug=["ships with an undocumented backdoor account", "contains an embedded static password"],
        components=["the telnet service", "the SSH daemon", "the device firmware", "the database connector"],
        metrics={"AV": {"N": 0.85, "A": 0.1, "L": 0.05}, "PR": {"N": 1.0},
                 "C": {"H": 0.95, "L": 0.05}, "I": {"H": 0.9, "L": 0.1}, "A": {"H": 0.85, "N": 0.15}},
        impact=["obtain full administrative access", "log in with root privileges", "take over the device"]),
    "info_disclosure": dict(
        weight=12, cwe="CWE-200",
        names=["information disclosure", "information exposure", "sensitive data exposure",
               "exposure of sensitive information to an unauthorized actor", "verbose error message"],
        bug=["includes internal data in error responses", "logs credentials in cleartext",
             "exposes a debug endpoint", "returns uninitialized memory"],
        components=["the debug endpoint", "the logging subsystem", "the error handler", "the status page",
                    "the API response serializer", "the kernel driver"],
        metrics={"AV": {"N": 0.65, "L": 0.3, "A": 0.05}, "PR": {"N": 0.45, "L": 0.45, "H": 0.10},
                 "C": {"L": 0.55, "H": 0.45}, "I": {"N": 1.0}, "A": {"N": 1.0}},
        impact=None),
    "dos": dict(
        weight=13, cwe="CWE-400",
        names=["denial of service", "uncontrolled resource consumption", "NULL pointer dereference",
               "infinite loop", "reachable assertion", "excessive memory allocation"],
        bug=["does not limit recursion depth", "enters an infinite loop on malformed input",
             "dereferences a NULL pointer", "allocates memory based on an attacker-controlled length"],
        components=["the HTTP/2 implementation", "the DNS resolver", "the JSON parser", "the TLS handshake code",
                    "the packet parser", "the regular expression engine"],
        metrics={"AV": {"N": 0.65, "L": 0.25, "A": 0.1}, "PR": {"N": 0.65, "L": 0.3, "H": 0.05},
                 "UI": {"N": 0.85, "R": 0.15},
                 "C": {"N": 1.0}, "I": {"N": 1.0}, "A": {"H": 0.75, "L": 0.25}},
        impact=None),
    "path_traversal": dict(
        weight=6, cwe="CWE-22",
        names=["directory traversal", "path traversal", "relative path traversal", "arbitrary file read"],
        bug=["does not sanitize '../' sequences in file names", "concatenates user input into file paths"],
        components=["the file download endpoint", "the attachment handler", "the theme loader", "the log viewer"],
        metrics={"AV": {"N": 0.95, "L": 0.05}, "PR": {"N": 0.5, "L": 0.4, "H": 0.1},
                 "C": {"H": 0.8, "L": 0.2}, "I": {"N": 0.55, "H": 0.3, "L": 0.15}, "A": {"N": 0.7, "H": 0.3}},
        impact=["read arbitrary files", "read arbitrary files on the server", "write files outside the web root"]),
    "privilege_escalation": dict(
        weight=7, cwe="CWE-269",
        names=["privilege escalation", "improper privilege management", "incorrect permission assignment",
               "insecure file permissions"],
        bug=["runs a helper binary with elevated privileges", "creates world-writable configuration files",
             "does not drop privileges correctly"],
        components=["the setuid helper", "the update service", "the kernel driver", "the installer",
                    "the system daemon"],
        metrics={"AV": {"L": 0.9, "N": 0.1}, "PR": {"L": 0.85, "N": 0.1, "H": 0.05},
                 "S": {"U": 0.8, "C": 0.2},
                 "C": {"H": 0.95, "L": 0.05}, "I": {"H": 0.95, "L": 0.05}, "A": {"H": 0.9, "N": 0.1}},
        impact=["gain root privileges", "escalate privileges to SYSTEM", "gain elevated privileges"]),
    "ssrf": dict(
        weight=3, cwe="CWE-918",
        names=["server-side request forgery (SSRF)", "SSRF"],
        bug=["fetches attacker-supplied URLs without restriction", "follows redirects to internal hosts"],
        components=["the webhook feature", "the URL preview service", "the image import function", "the PDF export"],
        metrics={"AV": {"N": 1.0}, "PR": {"N": 0.6, "L": 0.4}, "S": {"C": 0.5, "U": 0.5},
                 "C": {"L": 0.6, "H": 0.4}, "I": {"L": 0.6, "N": 0.3, "H": 0.1}, "A": {"N": 0.9, "L": 0.1}},
        impact=["send requests to internal services", "scan the internal network", "access cloud metadata endpoints"]),
    "open_redirect": dict(
        weight=3, cwe="CWE-601",
        names=["open redirect", "URL redirection to untrusted site", "open redirection"],
        bug=["does not validate the return URL parameter", "trusts the next parameter for redirects"],
        components=["the login page", "the logout handler", "the OAuth callback", "the link tracking service"],
        metrics={"AV": {"N": 1.0}, "PR": {"N": 1.0}, "UI": {"R": 1.0}, "S": {"C": 0.85, "U": 0.15},
                 "C": {"L": 0.85, "N": 0.15}, "I": {"L": 1.0}, "A": {"N": 1.0}},
        impact=["redirect users to arbitrary web sites", "conduct phishing attacks"]),
    "xxe": dict(
        weight=2, cwe="CWE-611",
        names=["XML external entity (XXE) injection", "XXE", "improper restriction of XML external entity reference"],
        bug=["enables external entity resolution in its XML parser", "processes DTDs from untrusted documents"],
        components=["the SAML parser", "the XML import feature", "the SOAP endpoint", "the document converter"],
        metrics={"AV": {"N": 0.95, "L": 0.05}, "PR": {"N": 0.6, "L": 0.4},
                 "C": {"H": 0.8, "L": 0.2}, "I": {"N": 0.8, "L": 0.2}, "A": {"N": 0.4, "L": 0.3, "H": 0.3}},
        impact=["read arbitrary files", "disclose internal files", "conduct SSRF attacks"]),
    "race_condition": dict(
        weight=3, cwe="CWE-362",
        names=["race condition", "time-of-check time-of-use (TOCTOU) race condition", "concurrent execution flaw"],
        bug=["checks file permissions before opening the file without locking", "shares state between threads unsafely"],
        components=["the temporary file handler", "the kernel driver", "the session manager", "the update service"],
        metrics={"AV": {"L": 0.7, "N": 0.3}, "AC": {"H": 0.95, "L": 0.05}, "PR": {"L": 0.7, "N": 0.3},
                 "C": {"H": 0.6, "L": 0.2, "N": 0.2}, "I": {"H": 0.6, "L": 0.2, "N": 0.2},
                 "A": {"H": 0.6, "N": 0.4}},
        impact=None),
    "physical_side_channel": dict(
        weight=2, cwe="CWE-1300",
        names=["side-channel information leak", "debug interface exposure", "improper protection of physical side channels",
               "unprotected JTAG interface", "lock screen bypass"],
        bug=["leaves the UART console enabled", "leaks key material through timing differences",
             "does not lock the debug port"],
        components=["the bootloader", "the hardware security module", "the lock screen", "the debug port"],
        metrics={"AV": {"P": 0.7, "L": 0.3}, "AC": {"H": 0.6, "L": 0.4}, "PR": {"N": 0.8, "L": 0.2},
                 "C": {"H": 0.5, "L": 0.5}, "I": {"N": 0.6, "L": 0.3, "H": 0.1}, "A": {"N": 0.8, "H": 0.2}},
        impact=None),
    "improper_input_validation": dict(
        weight=6, cwe="CWE-20",
        names=["improper input validation", "a logic error", "an unspecified vulnerability",
               "improper validation of specified quantity in input", "insufficient validation"],
        bug=["does not validate certain fields", "mishandles certain input", "contains a logic flaw"],
        components=["the API handler", "the configuration parser", "the message handler", "the import function",
                    "the core module"],
        metrics={},  # uses DEFAULT -> wide spread of severities, little class signal
        impact=None),
}

# ---------------------------------------------------------------------------
# 2. Phrase pools that verbalise individual CVSS metrics.
# ---------------------------------------------------------------------------
ACTOR = {  # keyed by (AV, PR)
    ("N", "N"): ["remote attackers", "unauthenticated remote attackers", "a remote unauthenticated attacker",
                 "remote attackers without authentication", "an unauthenticated attacker"],
    ("N", "L"): ["remote authenticated users", "authenticated attackers", "a low-privileged authenticated user",
                 "an authenticated remote attacker", "logged-in users"],
    ("N", "H"): ["remote authenticated administrators", "attackers with administrator privileges",
                 "a privileged remote user", "authenticated admin users"],
    ("A", "N"): ["adjacent network attackers", "attackers on the same network segment",
                 "an attacker within Bluetooth range", "attackers on the local Wi-Fi network"],
    ("A", "L"): ["authenticated attackers on the adjacent network", "authenticated users on the local network"],
    ("A", "H"): ["administrators on the adjacent network"],
    ("L", "N"): ["local attackers", "local users", "a local attacker"],
    ("L", "L"): ["local users", "local low-privileged users", "a local authenticated attacker", "unprivileged local users"],
    ("L", "H"): ["local administrators", "local users with administrative privileges"],
    ("P", "N"): ["physically proximate attackers", "an attacker with physical access",
                 "attackers with physical access to the device"],
    ("P", "L"): ["physically proximate authenticated users"],
    ("P", "H"): ["physically present administrators"],
}
GENERIC_ACTOR = ["attackers", "an attacker", "a malicious actor", "a threat actor"]

VECTOR_UI_NONE = ["via a crafted HTTP request", "via a crafted packet", "via a crafted request",
                  "via the {param} parameter", "via crafted input", "via a crafted API call",
                  "by sending specially crafted data"]
VECTOR_UI_REQ = ["via a crafted link", "by convincing a user to open a crafted file",
                 "via a malicious web page visited by the victim", "via a crafted document",
                 "when a victim clicks a crafted URL", "if a user opens a malicious attachment"]
PARAMS = ["id", "name", "q", "file", "page", "url", "user", "cmd", "path", "lang", "redirect", "sort"]
COND_AC_HIGH = [", if a non-default configuration is enabled", " under specific race conditions",
                ", although exploitation requires winning a race", " in certain non-default configurations",
                ", provided the attacker can perform a man-in-the-middle attack",
                ", but successful exploitation requires precise timing"]
SCOPE_CHANGED = [" affecting other components", " and escape the sandbox", " and break out of the container",
                 " beyond the vulnerable component"]

# Generic impact phrases keyed by the (C, I, A) impact profile.
def generic_impact(c, i, a, rng):
    if c == "H" and i == "H" and a == "H":
        pool = ["execute arbitrary code", "execute arbitrary code or cause a denial of service",
                "take complete control of the affected system", "compromise the system"]
    elif c == "H" and i == "H":
        pool = ["read and modify sensitive data", "gain unauthorized read and write access"]
    elif c == "H" and i != "H":
        pool = ["obtain sensitive information", "read sensitive memory contents", "disclose credentials",
                "access confidential information"]
    elif i == "H":
        pool = ["modify arbitrary files", "overwrite arbitrary files", "tamper with data"]
    elif a == "H" and c == "N" and i == "N":
        pool = ["cause a denial of service (crash)", "cause a denial of service (application crash)",
                "crash the service", "cause a denial of service via resource exhaustion"]
    elif a == "L" and c == "N" and i == "N":
        pool = ["cause a partial denial of service", "degrade performance", "cause a temporary service disruption"]
    elif c == "L" and i == "N":
        pool = ["obtain potentially sensitive information", "obtain limited information",
                "discover internal file paths", "enumerate valid usernames", "obtain version information"]
    else:
        pool = ["have an unspecified impact", "cause unexpected behavior", "affect integrity",
                "obtain or modify limited information"]
    return pool[rng.integers(len(pool))]


# Pseudo-word vendor/product names (no signal, no real vendors).
SYL = ["ax", "bel", "cor", "dra", "en", "fi", "gor", "hal", "ix", "jen", "kor", "lum", "mo", "nex",
       "or", "pra", "quo", "ril", "sen", "tal", "ur", "vex", "wyn", "zo"]
KINDS = ["Web Server", "CMS", "Router Firmware", "PDF Reader", "Mail Gateway", "VPN Client", "Media Player",
         "ERP Suite", "Database Server", "IoT Camera Firmware", "Build Server", "Mobile App", "Hypervisor",
         "Wiki", "Forum Software", "NAS Firmware", "Browser", "Office Suite", "PBX", "Monitoring Agent"]


def pseudo_name(rng):
    n = rng.integers(4, 6)  # 4-5 syllables -> ~8.3M distinct names, so names (almost) never repeat
    return "".join(SYL[rng.integers(len(SYL))] for _ in range(n)).capitalize()


def version_phrase(rng):
    a, b, c = rng.integers(0, 12), rng.integers(0, 20), rng.integers(0, 30)
    forms = [f"before {a}.{b}.{c}", f"through {a}.{b}.{c}", f"{a}.x before {a}.{b}.{c}",
             f"versions prior to {a}.{b}", f"{a}.{b} and earlier", f"up to and including {a}.{b}.{c}"]
    return forms[rng.integers(len(forms))]


def choose(rng, dist):
    """Sample a key from a {value: prob} dict."""
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def pick(rng, pool):
    return pool[rng.integers(len(pool))]


# ---------------------------------------------------------------------------
# 3. Record generation.
# ---------------------------------------------------------------------------
def sample_metrics(cls, rng):
    spec = {**DEFAULT, **CLASSES[cls]["metrics"]}
    return {k: choose(rng, spec[k]) for k in ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]}


def render(cls, m, rng):
    """Turn (class, CVSS vector) into a CVE-style description.

    Each metric is verbalised only with some probability, so the text is a
    noisy, incomplete view of the vector.
    """
    spec = CLASSES[cls]
    product = f"{pseudo_name(rng)} {pick(rng, KINDS)}"
    vendor = pseudo_name(rng)
    comp = pick(rng, spec["components"])
    name = pick(rng, spec["names"])
    vers = version_phrase(rng)

    # Actor phrase reveals AV and PR (80% of the time).
    actor = pick(rng, ACTOR[(m["AV"], m["PR"])]) if rng.random() < 0.80 else pick(rng, GENERIC_ACTOR)
    # Impact: class-specific phrase (if any) or CIA-driven phrase.
    if spec["impact"] is not None and rng.random() < 0.7:
        impact = pick(rng, spec["impact"])
    else:
        impact = generic_impact(m["C"], m["I"], m["A"], rng)
    if m["A"] == "H" and "denial of service" not in impact and rng.random() < 0.25:
        impact += " or cause a denial of service"
    # Attack vector / user interaction (75% revealed).
    if rng.random() < 0.75:
        vec = pick(rng, VECTOR_UI_REQ if m["UI"] == "R" else VECTOR_UI_NONE)
        vec = " " + vec.format(param=pick(rng, PARAMS))
    else:
        vec = ""
    cond = pick(rng, COND_AC_HIGH) if (m["AC"] == "H" and rng.random() < 0.7) else ""
    scope = pick(rng, SCOPE_CHANGED) if (m["S"] == "C" and cls not in ("xss", "open_redirect")
                                         and rng.random() < 0.5) else ""

    r = rng.random()
    if r < 0.50:   # classic NVD template
        text = f"{name[0].upper() + name[1:]} in {comp} in {vendor} {product} {vers} allows {actor} to {impact}{scope}{vec}{cond}."
    elif r < 0.72:  # "An issue was discovered" template
        text = (f"An issue was discovered in {vendor} {product} {vers}. {comp[0].upper() + comp[1:]} "
                f"{pick(rng, spec['bug'])}, which allows {actor} to {impact}{scope}{vec}{cond}.")
    elif r < 0.90:  # advisory-style template
        text = (f"{vendor} {product} {vers} is vulnerable to {name}. "
                f"{actor[0].upper() + actor[1:]} could exploit this vulnerability{vec} to {impact}{scope}{cond}.")
    else:          # vague, low-information template (~10%)
        vague = pick(rng, ["may lead to a security impact", "could compromise the affected product",
                           "has been addressed in the latest release", "may allow an attacker to affect the system"])
        text = f"A vulnerability in {comp} of {vendor} {product} {vers} {vague}."
    return text


def generate(n=config.N_SAMPLES, seed=config.SEED, label_noise=config.LABEL_NOISE):
    """Generate `n` labelled records as a DataFrame (zero-score vectors are resampled)."""
    rng = np.random.default_rng(seed)
    names = list(CLASSES)
    w = np.array([CLASSES[c]["weight"] for c in names], dtype=float)
    rows, seen = [], set()
    while len(rows) < n:
        cls = names[rng.choice(len(names), p=w / w.sum())]
        m = sample_metrics(cls, rng)
        text = render(cls, m, rng)
        # Analyst-disagreement noise: re-sample one metric AFTER the text was written.
        noisy = False
        if rng.random() < label_noise:
            k = pick(rng, ["PR", "UI", "C", "I", "A", "AC"])
            m = dict(m, **{k: choose(rng, DEFAULT[k])})
            noisy = True
        score = base_score(m)
        if score == 0.0 or text in seen:   # CVSS "None" or exact duplicate -> skip
            continue
        seen.add(text)
        rows.append(dict(cve_id=f"SYN-{2020 + len(rows) % 6}-{10000 + len(rows)}", description=text,
                         vuln_class=cls, cwe=CLASSES[cls]["cwe"], cvss_vector=vector_string(m),
                         base_score=score, severity=severity(score), label_noise=noisy))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(config.DATASET_CSV, index=False)
    print(df["severity"].value_counts(normalize=True).reindex(config.LABELS).round(3))
    print(df.sample(5, random_state=0)[["description", "severity"]].to_string())
