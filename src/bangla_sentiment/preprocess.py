"""Light, reversible-free text normalisation for Bangla reviews.

Deliberately simple: the goal is to remove noise that hurts bag-of-n-gram models
(links, invisible characters, digit variants, repeated punctuation) without
stemming or dictionary lookups that would need extra Bangla NLP libraries.
"""

from __future__ import annotations

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_INVISIBLE = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_REPEAT_PUNCT = re.compile(r"([!?.,।\-])\1{1,}")
_SPACES = re.compile(r"\s+")


def normalize(text: object) -> str:
    """Return a cleaned version of ``text``. Non-strings become an empty string."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = _URL.sub(" ", text)
    text = text.translate(_BANGLA_DIGITS)
    text = text.lower()  # affects Latin letters only (code-mixed reviews)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip()
