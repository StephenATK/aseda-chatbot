"""
textprep.py
===========
Group 13 - shared text-cleaning and preprocessing functions.

This ONE file is used by both the Colab notebook (training) and the Streamlit
app (live intent routing). Keeping it in one place guarantees that a message
typed into the app is cleaned exactly the same way as the training data was,
which fixes the train/serve mismatch in the earlier version (the app used to
send raw lower-case text to a model trained on stemmed, stopword-free text).

Two cleaning tracks are used on purpose:

  * light_clean()  -> for the Llama generator. Keeps case and punctuation,
                      because a generative model must learn natural writing.
  * router_prep()  -> for the classical intent router (TF-IDF + NB / SVM).
                      Regex clean -> lowercase -> stopword removal -> stemming.
"""

import re
import unicodedata

from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# Regular expressions (Stage 1: data cleaning)
# ---------------------------------------------------------------------------
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_MULTISPACE = re.compile(r"\s+")
RE_NON_ALNUM = re.compile(r"[^a-zA-Z0-9{}\s]")       # keep {{ }} slot braces
RE_REPEATED_CH = re.compile(r"([a-zA-Z])\1{2,}")      # soooo -> soo
RE_SLOT = re.compile(r"\{\{\s*(.*?)\s*\}\}")          # {{ Order Number }}
RE_SLOT_TOKEN = re.compile(r"__slot\d+__")

# Named rules, so the notebook can report how often each one fires.
REGEX_RULES = {
    "url": RE_URL,
    "email": RE_EMAIL,
    "slot_placeholder": RE_SLOT,
    "elongated_word": RE_REPEATED_CH,
    "punctuation_symbol": RE_NON_ALNUM,
    "extra_whitespace": re.compile(r"\s{2,}|^\s|\s$"),
}

# Bitext answers use {{Slot}} placeholders such as {{Order Number}}. They are
# KEPT, not removed: the chatbot should learn to write a placeholder instead of
# inventing an order number (this is how a No-RAG bot avoids hallucinating).


def extract_slots(text):
    """Return slot names found in a string, e.g. ['Order Number']."""
    return [s.strip() for s in RE_SLOT.findall(str(text))]


def normalise_slots(text):
    """'{{ Order  Number }}' -> '{{Order Number}}'."""
    return RE_SLOT.sub(lambda m: "{{" + RE_MULTISPACE.sub(" ", m.group(1)).strip() + "}}", text)


def light_clean(text):
    """Track A - for the Llama generator. Unicode + whitespace + slot spacing only."""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = normalise_slots(text)
    # tidy spaces but keep line breaks, which Bitext answers use for steps
    text = "\n".join(RE_MULTISPACE.sub(" ", line).strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def regex_clean(text):
    """Track B, step 1 - rule-based cleaning for the router."""
    text = light_clean(text)
    text = RE_URL.sub(" urltoken ", text)
    text = RE_EMAIL.sub(" emailtoken ", text)
    text = RE_REPEATED_CH.sub(r"\1\1", text)
    # expand negative contractions so the apostrophe stripper keeps "not"
    text = re.sub(r"(?i)\bwon't\b", "will not", text)
    text = re.sub(r"(?i)\bcan't\b", "can not", text)
    text = re.sub(r"(?i)n't\b", " not", text)
    # protect {{slots}} from the punctuation stripper with letters-only markers
    slots = []

    def _hold(m):
        slots.append(m.group(1))
        return " slotmarker%s " % chr(97 + len(slots) - 1)

    text = RE_SLOT.sub(_hold, text)
    text = RE_NON_ALNUM.sub(" ", text)
    for i, s in enumerate(slots):
        text = text.replace("slotmarker%s" % chr(97 + i), "{{" + s.replace(" ", "_") + "}}")
    return RE_MULTISPACE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Stage 2: preprocessing (tokenise, lowercase, stopwords, stemming)
# ---------------------------------------------------------------------------
# Curated English stopword list. Negation words (not, no, don't, can't ...) are
# deliberately NOT removed: "I don't want to cancel" and "I want to cancel"
# should not look identical to the router.
NEGATIONS = {
    "no", "nor", "not", "don't", "doesn't", "didn't", "can't", "cannot",
    "couldn't", "won't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "mustn't", "shan't",
}

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during each
few for from further had has have having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is it
it's its itself let's me more most my myself of off on once only or other ought
our ours ourselves out over own same she she'd she'll she's should so some such
than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up very
was we we'd we'll we're we've were what what's when when's where where's which
while who who's whom why why's with would you you'd you'll you're you've your
yours yourself yourselves
""".split()) - NEGATIONS

_stemmer = PorterStemmer()


def preprocess(text, remove_stopwords=True, stem=True):
    tokens = []
    for tok in text.lower().split():
        if tok.startswith("{{"):
            tokens.append("slottoken")
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        tokens.append(_stemmer.stem(tok) if stem else tok)
    return " ".join(tokens)


def router_text(text):
    """Full Track B pipeline for one message."""
    return preprocess(regex_clean(text))


def router_prep(texts):
    """Vectorised version used inside the saved sklearn Pipeline."""
    return [router_text(t) for t in texts]


# ---------------------------------------------------------------------------
# Hallucination check (used by the notebook's evaluation)
# ---------------------------------------------------------------------------
# A No-RAG bot has no access to the customer's account, so any concrete detail
# it states (an order number, a phone number, a price, a link, a policy time
# window) that the customer did not supply is invented.
SPECIFIC_PATTERNS = {
    "url": re.compile(r"https?://\S+|www\.\S+"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[a-z]{2,}\b", re.I),
    "phone": re.compile(r"(?<!\w)\+?\d[\d\-\s().]{7,}\d(?!\w)"),
    "long_number": re.compile(r"(?<![\w{])\d{4,}(?![\w}])"),
    "money": re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|eur|gbp|dollars)\b", re.I),
    "time_window": re.compile(r"\b\d+\s?(?:-|to)?\s?\d*\s?(?:business\s|working\s)?(?:days?|hours?|weeks?|minutes?)\b", re.I),
}


def invented_specifics(answer, question):
    """Return {type: [values]} for concrete details in `answer` that do not
    appear in `question`. {{slot}} placeholders are removed first, because a
    placeholder is the correct way NOT to invent a value."""
    answer_wo_slots = RE_SLOT.sub(" ", str(answer))
    q = str(question).lower()
    q_digits = re.sub(r"\D", "", q)
    found = {}
    for kind, pat in SPECIFIC_PATTERNS.items():
        for val in pat.findall(answer_wo_slots):
            val = val.strip().rstrip(".,;:!?)")
            v = val.lower()
            digits = re.sub(r"\D", "", v)
            grounded = v in q or (len(digits) >= 4 and digits in q_digits)
            if not grounded:
                found.setdefault(kind, []).append(val)
        # blank out what this pattern matched so e.g. a phone number is not
        # counted again as a "long number"
        answer_wo_slots = pat.sub(" ", answer_wo_slots)
    return found
