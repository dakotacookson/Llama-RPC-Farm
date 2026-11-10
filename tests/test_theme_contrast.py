"""WCAG contrast gate for every theme (dark + light variants).

Thresholds (WCAG 2.1 AA):
- primary/secondary text on bg/surface/raised: 4.5:1
- muted/accent/semantic colors on bg/surface: 3:1 (large text / UI)
"""
import os
import re

STYLE = os.path.join(os.path.dirname(__file__), "..", "ui", "static", "style.css")

TEXT_PAIRS = [("--text", b) for b in ("--bg", "--surface", "--raised")] + \
             [("--text-2", b) for b in ("--bg", "--surface", "--raised")]
UI_PAIRS = [(fg, b) for fg in ("--muted", "--accent", "--ok", "--warn", "--bad")
            for b in ("--bg", "--surface")]


def _rgb(value):
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _luminance(value):
    def channel(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(value)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _ratio(fg, bg):
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _blocks():
    css = open(STYLE, encoding="utf-8").read()
    for match in re.finditer(r':root[^{]*\{([^}]*)\}', css):
        body = match.group(1)
        if "--bg:" not in body:
            continue
        name_match = re.search(r'data-theme="([^"]+)"', match.group(0))
        name = name_match.group(1) if name_match else "default"
        values = {k: v.strip() for k, v in
                  re.findall(r'(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6});', body)}
        if values:
            yield name, values


def test_every_theme_meets_contrast_standard():
    failures = []
    for name, values in _blocks():
        for fg, bg in TEXT_PAIRS:
            if fg in values and bg in values:
                ratio = _ratio(values[fg], values[bg])
                if ratio < 4.5:
                    failures.append(
                        f"{name}: {fg} on {bg} = {ratio:.2f} (< 4.5)")
        for fg, bg in UI_PAIRS:
            if fg in values and bg in values:
                ratio = _ratio(values[fg], values[bg])
                if ratio < 3.0:
                    failures.append(
                        f"{name}: {fg} on {bg} = {ratio:.2f} (< 3.0)")
    assert not failures, "\n".join(failures)
