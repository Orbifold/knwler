"""
Language / i18n support.
"""

import json
from pathlib import Path

from knwler.config import PACKAGE_ROOT, console

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
LANGUAGES_FILE = PACKAGE_ROOT / "languages.json"
DEFAULT_LANGUAGE = "en"
_LANGUAGES: dict = {}
_CURRENT_LANG: str = DEFAULT_LANGUAGE


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def load_languages() -> dict:
    """Load language definitions from JSON file."""
    global _LANGUAGES
    if not _LANGUAGES:
        if LANGUAGES_FILE.exists():
            _LANGUAGES = json.loads(LANGUAGES_FILE.read_text(encoding="utf-8"))
        else:
            console.print(
                f"[yellow]Warning: {LANGUAGES_FILE} not found, using English[/yellow]"
            )
            _LANGUAGES = {
                "en": {"name": "English", "prompts": {}, "ui": {}, "console": {}}
            }
    return _LANGUAGES


def get_lang() -> dict:
    """Get the current language dictionary."""
    langs = load_languages()
    return langs.get(_CURRENT_LANG, langs.get(DEFAULT_LANGUAGE, {}))


def get_prompt(key: str, **kwargs) -> str:
    """Get a localized prompt template, formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("prompts", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("prompts", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_ui(key: str, **kwargs) -> str:
    """Get a localized UI string, formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("ui", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("ui", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def get_console_msg(key: str, **kwargs) -> str:
    """Get a localized console message, formatted with *kwargs*."""
    lang = get_lang()
    template = lang.get("console", {}).get(key, "")
    if not template:
        template = load_languages().get("en", {}).get("console", {}).get(key, "")
    return template.format(**kwargs) if template else ""


def set_language(lang_code: str):
    """Set the current language."""
    global _CURRENT_LANG
    langs = load_languages()
    if lang_code in langs:
        _CURRENT_LANG = lang_code
    else:
        console.print(
            f"[yellow]Language '{lang_code}' not found, using English[/yellow]"
        )
        _CURRENT_LANG = DEFAULT_LANGUAGE


def get_current_language() -> str:
    """Return the current language code."""
    return _CURRENT_LANG
