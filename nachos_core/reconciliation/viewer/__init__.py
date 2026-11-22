"""Offline reconciliation viewer: one self-contained, CSP-hashed HTML document."""

import base64
import hashlib
from importlib import resources

__all__ = ["viewer_html"]

_ASSETS = ("viewer.html", "viewer.css", "core.js", "ui.js")


def _read(name: str) -> str:
    return (resources.files(__package__) / name).read_bytes().decode("utf-8")


def _script_safe(source: str) -> str:
    # Equivalent inside JS strings/comments; prevents premature script termination.
    return source.replace("</script", "<\\/script").replace("<!--", "<\\!--")


def _hash(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def viewer_html() -> bytes:
    """Return the deterministic, network-free viewer document as UTF-8 bytes."""
    template, css, core, ui = (_read(n) for n in _ASSETS)
    script = "\n" + _script_safe(core) + "\n" + _script_safe(ui) + "\n"
    # Core and UI share one inline block so the CSP needs a single script hash.
    style = "\n" + css + "\n"
    csp = (
        "default-src 'none'; script-src "
        + _hash(script)
        + "; style-src "
        + _hash(style)
        + "; connect-src 'none'; img-src 'none'; object-src 'none'; "
        "base-uri 'none'; form-action 'none'"
    )
    page = (
        template.replace("@@CSP@@", csp).replace("@@STYLE@@", style).replace("@@SCRIPT@@", script)
    )
    return page.encode("utf-8")
