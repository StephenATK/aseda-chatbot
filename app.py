"""
Group 13 - Llama-Based Customer-Service Chatbot (No RAG)
MSBA 610 Advanced Text Analytics

Streamlit app. Every number, chart and answer shown here is read from the
`artifacts/` folder produced by the Colab notebook (Group13_Chatbot_Colab.ipynb).
Nothing is hard-coded, and a missing file shows a clear message instead of an error.

Run locally:   streamlit run app.py
"""

import json
import math
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import textprep  # same cleaning code the notebook used for training

st.set_page_config(page_title="Aseda — Customer Service", page_icon="💬", layout="wide")

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

C_BASE, C_FT, C_REF = "#94A3B8", "#0F766E", "#C2410C"
C_NB, C_SVM = "#B45309", "#1D4ED8"
MODEL_COLORS = alt.Scale(domain=["Base", "Fine-tuned", "Reference"], range=[C_BASE, C_FT, C_REF])
ROUTER_COLORS = alt.Scale(domain=["Naive Bayes", "Linear SVM"], range=[C_NB, C_SVM])
ROUTER_SHORT = {"nb": "Naive Bayes", "svm": "Linear SVM"}
YAX = alt.Axis(labelOverlap=False, labelLimit=240)  # show every category label

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -0.01em; color: #0B1F33; }
.block-container { padding-top: 2.2rem; max-width: 1250px; }
.hero { background: linear-gradient(120deg, #0B1F33 0%, #0F4C5C 55%, #0F766E 100%); color: #F8FAFC;
        border-radius: 18px; padding: 30px 34px; margin-bottom: 18px; }
.hero h1 { color: #F8FAFC !important; font-size: 2.15rem; margin: 0 0 6px 0; }
.hero p { color: #CBD5E1; margin: 0; font-size: 1.02rem; }
.tag { display: inline-block; background: rgba(255,255,255,.12); color: #E2E8F0; border-radius: 999px;
       padding: 3px 12px; font-size: .78rem; margin: 0 6px 12px 0; letter-spacing: .02em; }
.kpi { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; height: 100%; }
.kpi .v { font-family: 'Plus Jakarta Sans', sans-serif; font-size: 1.7rem; font-weight: 800; color: #0B1F33; line-height: 1.1; }
.kpi .l { color: #64748B; font-size: .82rem; margin-top: 4px; }
.kpi .d { font-size: .82rem; margin-top: 6px; font-weight: 600; }
.up { color: #0F766E; } .down { color: #B91C1C; } .flat { color: #64748B; }
.flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 8px 0 4px 0; }
.step { background: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 12px; padding: 10px 14px; font-size: .9rem; }
.step b { display: block; color: #0B1F33; font-size: .92rem; }
.step span { color: #64748B; font-size: .78rem; }
.step.hl { background: #ECFDF5; border-color: #99F6E4; }
.arrow { color: #94A3B8; font-size: 1.2rem; }
.card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; margin-bottom: 12px; }
.card h4 { margin: 0 0 8px 0; font-size: 1rem; }
.answer { font-size: .9rem; line-height: 1.55; color: #1E293B; }
.badge { display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: .75rem; font-weight: 600; margin-right: 6px; }
.b-ft { background: #CCFBF1; color: #115E59; } .b-base { background: #E2E8F0; color: #334155; }
.b-ref { background: #FFEDD5; color: #9A3412; } .b-warn { background: #FEE2E2; color: #991B1B; }
.b-ok { background: #DCFCE7; color: #166534; } .b-ns { background: #F1F5F9; color: #64748B; }
.chip { display: inline-block; background: #EEF2FF; color: #3730A3; border-radius: 8px; padding: 2px 8px;
        margin: 2px; font-family: ui-monospace, monospace; font-size: .8rem; }
.note { color: #64748B; font-size: .85rem; }
mark { background: #FECACA; padding: 0 2px; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Loading helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(name):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = ART / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_router(key):
    p = ART / f"router_{key}.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # usually a scikit-learn version mismatch
        st.error(f"Could not load `{p.name}`: {e}. Check that requirements.txt pins the scikit-learn version from the notebook.")
        return None


@st.cache_resource(show_spinner=False)
def load_word_vectors():
    p = ART / "lsa_word_vectors.npz"
    if not p.exists():
        return None, None
    z = np.load(p, allow_pickle=True)
    return z["vocab"], z["vectors"].astype(np.float32)


def need(obj, what):
    if obj is None:
        st.info(f"**{what}** isn't available yet. Run the Colab notebook, unzip `group13_deploy.zip` into this "
                "folder (so `artifacts/` sits next to `app.py`), and reload.")
        return False
    return True


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpi(col, value, label, delta=None, good=None):
    cls = "flat" if good is None else ("up" if good else "down")
    d = f'<div class="d {cls}">{delta}</div>' if delta else ""
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div>{d}</div>', unsafe_allow_html=True)


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def html_text(t):
    """Escape text and keep its line breaks when shown inside an HTML block."""
    return esc(t).replace("\n", "<br>")


def metric(summary, key):
    return next((m for m in summary["metrics"] if m["key"] == key), None) if summary else None


def fmt_metric(key, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if key == "log_perplexity":
        return f"{math.exp(v):.1f}"
    if key in ("hallucination", "uses_placeholder"):
        return f"{100 * v:.1f}%"
    return f"{v:.3f}"


cfg = load_json("config.json")
summary = load_json("llm_summary.json")
data_summary = load_json("data_summary.json")
router_cmp = load_csv("router_comparison.csv")
gens = load_csv("generations.csv")
MODEL_NAME = (cfg or {}).get("model_used", "Llama-3.2-3B-Instruct").split("/")[-1]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
PAGES = ["Overview", "Chat demo", "1 · Data cleaning", "2 · Exploration", "3 · Text representation",
         "4 · Intent router", "5 · Llama: base vs fine-tuned", "6 · Conclusions"]
with st.sidebar:
    st.markdown("### 💬 Aseda")
    st.caption("Group 13 · Llama-based customer-service chatbot · No RAG")
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.divider()
    if cfg:
        st.caption(f"Model: `{MODEL_NAME}`  \nRun: {cfg.get('created', '')}  \nGPU: {cfg.get('gpu', '')}")
    else:
        st.caption("No results loaded yet.")


# ============================================================================
# Overview
# ============================================================================
def page_overview():
    st.markdown(
        f"""
<div class="hero">
  <span class="tag">MSBA 610 · Advanced Text Analytics</span><span class="tag">Group 13</span>
  <h1>Aseda: a customer-service chatbot that answers without retrieval</h1>
  <p>{esc(MODEL_NAME)} answers from its own weights only (no RAG). A classical intent router tells it what the
  customer wants, and QLoRA fine-tuning on the Bitext customer-support dataset teaches it how a support agent replies.</p>
</div>""",
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    if data_summary:
        kpi(c[0], f"{data_summary['rows_final']:,}", "cleaned customer questions")
        kpi(c[1], data_summary["n_intents"], f"intents in {data_summary['n_categories']} categories")
    if router_cmp is not None:
        best = router_cmp.sort_values("f1_macro", ascending=False).iloc[0]
        kpi(c[2], f"{best.f1_macro:.3f}", f"router macro-F1 ({ROUTER_SHORT.get(best.key, best.key)})")
    if summary:
        bs, h = metric(summary, "bertscore"), metric(summary, "hallucination")
        kpi(c[3], f"{bs['finetuned']:.3f}", "BERTScore, fine-tuned", f"base {bs['base']:.3f} · Δ {bs['diff_ft_minus_base']:+.3f}",
            bs["diff_ft_minus_base"] > 0)
        kpi(c[4], f"{100 * h['finetuned']:.1f}%", "answers with invented details", f"base {100 * h['base']:.1f}%",
            h["finetuned"] <= h["base"])

    st.markdown("#### How a message flows through the system")
    router_name = ROUTER_SHORT.get((cfg or {}).get("best_router", "svm"), "Linear SVM")
    st.markdown(
        f"""
<div class="flow">
 <div class="step"><b>Customer message</b><span>free text</span></div><div class="arrow">→</div>
 <div class="step"><b>Cleaning</b><span>regex · stopwords · stemming</span></div><div class="arrow">→</div>
 <div class="step"><b>Intent router</b><span>TF-IDF + {router_name}</span></div><div class="arrow">→</div>
 <div class="step hl"><b>{esc(MODEL_NAME)} + LoRA</b><span>system prompt + routing note</span></div><div class="arrow">→</div>
 <div class="step"><b>Answer</b><span>placeholders, never invented details</span></div>
</div>
<p class="note">No documents are retrieved at any point. Account-specific details are written as placeholders such as
<code>{{{{Order Number}}}}</code>, so the bot never makes them up.</p>""",
        unsafe_allow_html=True,
    )

    if summary:
        st.markdown("#### Headline findings")
        for line in findings():
            st.markdown(f"- {line}")
    if cfg:
        with st.expander("Experiment configuration"):
            st.json({k: v for k, v in cfg.items() if k not in ("system_prompt",)})
            st.caption("System prompt (same for both models)")
            st.code(cfg.get("system_prompt", ""), language="text")


def findings():
    out = []
    best = summary["models"][summary["best_model"]]["label"]
    w = summary.get("significant_wins", {})
    out.append(f"**Best generator: {best}.** It is significantly better on {w.get(summary['best_model'], 0)} of "
               f"{len(summary['metrics'])} metrics (paired test, p < 0.05).")
    sig = [m for m in summary["metrics"] if m["significant"] and m["better"] == "finetuned"]
    if sig:
        out.append("Fine-tuning significantly improved: " + ", ".join(
            f"{m['label']} ({fmt_metric(m['key'], m['base'])} → {fmt_metric(m['key'], m['finetuned'])})" for m in sig) + ".")
    worse = [m for m in summary["metrics"] if m["significant"] and m["better"] == "base"]
    if worse:
        out.append("The base model stayed ahead on: " + ", ".join(
            f"{m['label']} ({fmt_metric(m['key'], m['base'])} vs {fmt_metric(m['key'], m['finetuned'])})" for m in worse) + ".")
    ppl = summary.get("reference_perplexity", {})
    if ppl.get("base") and ppl.get("finetuned"):
        out.append(f"Perplexity of the human-written reference answers fell from {ppl['base']:.1f} to {ppl['finetuned']:.1f}: "
                   "the fine-tuned model finds the target support style far more predictable.")
    if router_cmp is not None:
        b = router_cmp.sort_values("f1_macro", ascending=False).iloc[0]
        out.append(f"The {ROUTER_SHORT.get(b.key)} router reaches macro-F1 {b.f1_macro:.3f} on held-out questions, "
                   f"so its routing note to Llama is almost always correct.")
    return out


# ============================================================================
# Chat demo
# ============================================================================
@st.cache_resource(show_spinner=False)
def chat_index():
    if gens is None:
        return None
    key = (cfg or {}).get("best_router", "svm")
    r = load_router(key)
    if r is None:
        return None
    tfidf = r.named_steps["tfidf"]
    M = tfidf.transform(textprep.router_prep(gens.question.tolist()))
    return r, tfidf, M


def router_top3(router, text):
    clf = router.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        s = router.predict_proba([text])[0]
        label = "probability"
    else:
        d = router.decision_function([text])[0]
        s = np.exp(d - d.max()); s = s / s.sum()
        label = "relative score"
    order = np.argsort(-s)[:3]
    return [(clf.classes_[i], float(s[i])) for i in order], label


def page_chat():
    st.markdown("## 💬 Chat with Aseda")
    st.markdown(
        '<p class="note">The <b>intent router runs live</b> on whatever you type. The Llama answers are '
        "<b>real answers the models gave to held-out test questions</b> in Colab. Llama is too large to run on free Streamlit "
        "hosting, so the demo shows the answer to the closest recorded question and says how close it is.</p>",
        unsafe_allow_html=True,
    )
    idx = chat_index()
    if not need(idx, "Recorded answers and the router"):
        return
    router, tfidf, M = idx

    c1, c2 = st.columns([3, 2])
    which = c1.radio("Answer from", ["Fine-tuned", "Base"], horizontal=True)
    if c2.button("Clear chat"):
        st.session_state.chat = []
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("suggest", gens.sample(4, random_state=7).question.tolist())

    if not st.session_state.chat:
        st.caption("Try one of these test questions, or type your own:")
        cols = st.columns(4)
        for i, q in enumerate(st.session_state.suggest):
            if cols[i].button(q[:70] + ("…" if len(q) > 70 else ""), key=f"sug{i}"):
                st.session_state.pending = q
                st.rerun()

    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "💬"):
            st.markdown(m["content"], unsafe_allow_html=True)

    text = st.chat_input("Describe your issue, e.g. 'I want to cancel my order'")
    text = text or st.session_state.pop("pending", None)
    if not text:
        return
    st.session_state.chat.append({"role": "user", "content": esc(text)})

    top3, score_label = router_top3(router, text)
    intent = top3[0][0]
    v = tfidf.transform([textprep.router_text(text)])
    sims = (M @ v.T).toarray().ravel()
    same = (gens.predicted_intent == intent).values
    pick = int(np.argmax(np.where(same, sims, -1))) if same.any() else int(np.argmax(sims))
    sim = float(sims[pick])
    row = gens.iloc[pick]
    col = "finetuned_answer" if which == "Fine-tuned" else "base_answer"
    badge = "b-ft" if which == "Fine-tuned" else "b-base"

    if sim >= 0.5:
        match = f'<span class="badge b-ok">close match · similarity {sim:.2f}</span>'
    else:
        match = (f'<span class="badge b-warn">no close recorded question · similarity {sim:.2f}</span>'
                 f"<br><span class='note'>Showing a recorded answer for another <code>{esc(intent)}</code> question.</span>")
    route = " ".join(f'<span class="chip">{esc(i)} {s:.0%}</span>' for i, s in top3)
    reply = (f'<b>Aseda</b> <span class="badge {badge}">{which} model</span>{match}<br>'
             f'<div class="answer" style="margin-top:8px">{html_text(row[col])}</div>'
             f'<hr style="margin:10px 0"><span class="note">Router ({score_label}): {route}<br>'
             f'Recorded test question: “{esc(row.question)}”</span>')
    st.session_state.chat.append({"role": "assistant", "content": reply})
    st.rerun()


# ============================================================================
# 1 · Data cleaning
# ============================================================================
def page_data():
    st.markdown("## 1 · Data cleaning and regular expressions")
    if not need(data_summary, "Data summary"):
        return
    d = data_summary
    c = st.columns(4)
    kpi(c[0], f"{d['rows_raw']:,}", "raw rows (Bitext)")
    kpi(c[1], f"{d['duplicate_pairs_removed']:,}", "duplicate question/answer pairs removed")
    kpi(c[2], f"{d['repeated_questions_removed']:,}", "repeated questions removed")
    kpi(c[3], f"{d['rows_final']:,}", "unique questions kept")
    st.markdown(
        """
Repeated questions are kept **once**, so an identical question can never appear in both the training and the
test set (which would inflate the router's accuracy). One stratified 80/10/10 split by intent is then made and
reused by every model.

**Two cleaning tracks**, because the two kinds of model need different text:

| | Track A – light clean | Track B – router text |
|---|---|---|
| Used by | Llama (generation) | Naive Bayes / Linear SVM (intent router) |
| Steps | unicode, quotes, whitespace, `{{slot}}` spacing | Track A + URL/e-mail tokens, elongated words, punctuation, lower-case, stopwords (negations kept), Porter stemming |
| Why | a generator must learn natural casing and punctuation | a bag-of-words model needs a compact, normalised vocabulary |

`{{Slot}}` placeholders (e.g. `{{Order Number}}`) are **kept**. A No-RAG bot can't see the customer's account,
so the right behaviour is to write the placeholder rather than invent a value.
        """
    )
    hits = load_csv("regex_rule_hits.csv")
    if hits is not None:
        st.markdown("#### How often each regular-expression rule fires on the raw text")
        ch = alt.Chart(hits).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("rule:N", sort="-x", title=None, axis=YAX), x=alt.X("pct_rows:Q", title="% of rows matched"),
            color=alt.Color("field:N", scale=alt.Scale(range=["#0F766E", "#94A3B8"]), title="field"),
            yOffset="field:N", tooltip=["field", "rule", "rows_matched", "pct_rows"]).properties(height=300)
        show_chart(ch)
    ex = load_csv("cleaning_examples.csv")
    if ex is not None:
        st.markdown("#### Before and after")
        show_df(ex, height=420)


# ============================================================================
# 2 · Exploration
# ============================================================================
def page_eda():
    st.markdown("## 2 · Exploratory text analytics")
    eda = load_json("eda_stats.json")
    if not need(eda, "EDA statistics"):
        return
    t1, t2, t3, t4, t5 = st.tabs(["Categories & intents", "Language variation", "Lengths", "Vocabulary", "Placeholders"])
    with t1:
        cat = pd.DataFrame({"category": list(eda["category_counts"]), "count": list(eda["category_counts"].values())})
        show_chart(alt.Chart(cat).mark_bar(cornerRadiusEnd=4, color=C_FT).encode(
            y=alt.Y("category:N", sort="-x", title=None, axis=YAX), x=alt.X("count:Q", title="questions"), tooltip=["category", "count"])
            .properties(height=320, title="Questions per category"))
        it = pd.DataFrame(eda["intent_counts"])
        show_chart(alt.Chart(it).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("intent:N", sort="-x", title=None, axis=YAX), x=alt.X("count:Q", title="questions"),
            color=alt.Color("category:N", legend=alt.Legend(orient="bottom", columns=6)), tooltip=["intent", "category", "count"])
            .properties(height=620, title="Questions per intent (coloured by category)"))
        st.caption("Classes are fairly balanced, so accuracy is not misleading, but macro-F1 is still reported.")
    with t2:
        fl = pd.DataFrame(eda["flag_counts"])
        if len(fl):
            fl["label"] = fl.flag + " · " + fl.meaning
            show_chart(alt.Chart(fl).mark_bar(cornerRadiusEnd=4, color="#6366F1").encode(
                y=alt.Y("label:N", sort="-x", title=None, axis=YAX), x=alt.X("count:Q", title="questions with this tag"),
                tooltip=["flag", "meaning", "count"]).properties(height=360, title="Bitext language-variation tags"))
            st.caption("Bitext tags each question with the variation it contains (letter meanings from the dataset card). "
                       "Colloquial wording, typos and keyword-only questions are what make routing and answering hard.")
    with t3:
        c = st.columns(2)
        for col, k, lab in [(c[0], "instruction_words", "Question length (words)"), (c[1], "response_words", "Answer length (words)")]:
            s = eda[k]
            col.markdown(f"**{lab}** · mean {s['mean']} · median {s['median']} · 95th pct {s['p95']} · max {s['max']}")
        ls = load_csv("length_sample.csv")
        if ls is not None:
            long = ls.melt(id_vars=["category"], value_vars=["instr_words", "resp_words"], var_name="field", value_name="words")
            long["field"] = long.field.map({"instr_words": "question", "resp_words": "answer"})
            long["words"] = long.words.clip(upper=400)
            show_chart(alt.Chart(long).mark_bar(opacity=.75).encode(
                x=alt.X("words:Q", bin=alt.Bin(maxbins=60), title="words (answers capped at 400)"),
                y=alt.Y("count():Q", stack=None, title="rows"), color=alt.Color("field:N", scale=alt.Scale(range=[C_REF, C_FT])))
                .properties(height=300))
            st.caption("Answers are about ten times longer than questions. This sets the generation length limit.")
    with t4:
        c = st.columns(2)
        for col, k, t in [(c[0], "top_unigrams", "Top words (router text, stemmed)"), (c[1], "top_bigrams", "Top two-word phrases")]:
            dfw = pd.DataFrame(eda[k])
            col.altair_chart(alt.Chart(dfw).mark_bar(cornerRadiusEnd=3, color=C_SVM).encode(
                y=alt.Y("term:N", sort="-x", title=None, axis=YAX), x=alt.X("count:Q"), tooltip=["term", "count"]).properties(height=520, title=t))
        st.markdown("**Most frequent words per category**")
        show_df(pd.DataFrame([{"category": k, "top words": ", ".join(v)} for k, v in eda["top_tokens_per_category"].items()]))
    with t5:
        c = st.columns(2)
        kpi(c[0], f"{eda['pct_responses_with_slots']}%", "of answers contain a {{placeholder}}")
        kpi(c[1], f"{eda['pct_instructions_with_slots']}%", "of questions contain a {{placeholder}}")
        rs = pd.DataFrame(eda["top_response_slots"])
        if len(rs):
            show_chart(alt.Chart(rs).mark_bar(cornerRadiusEnd=3, color=C_REF).encode(
                y=alt.Y("slot:N", sort="-x", title=None, axis=YAX), x="count:Q", tooltip=["slot", "count"])
                .properties(height=420, title="Most common placeholders in answers"))


# ============================================================================
# 3 · Representation
# ============================================================================
def page_repr():
    st.markdown("## 3 · Text representation")
    rep = load_json("representation.json")
    if not need(rep, "Representation results"):
        return
    st.markdown(
        """
Three representations are compared:
* **TF-IDF** (unigrams and bigrams): sparse and interpretable, used by the intent router.
* **LSA**, a 128-dimensional Truncated SVD of TF-IDF: dense word and document vectors learned from this corpus.
* **MiniLM sentence embeddings** (384-d, pretrained): contextual meaning, also used for the relevance metrics.
        """
    )
    c = st.columns(4)
    kpi(c[0], f"{100 * rep['lsa_explained_variance']:.1f}%", "TF-IDF variance kept by 128-d LSA")
    kpi(c[1], f"{rep['silhouette_intent_lsa']:.3f}", "intent separation · LSA (silhouette)")
    kpi(c[2], f"{rep['silhouette_intent_minilm']:.3f}", "intent separation · MiniLM (silhouette)")
    better = "MiniLM" if rep["silhouette_intent_minilm"] > rep["silhouette_intent_lsa"] else "LSA"
    kpi(c[3], better, "separates intents better")
    st.caption("Silhouette ranges from −1 to 1. Higher means questions with the same intent sit closer together than questions with other intents.")

    emb = load_csv("embedding_map.csv")
    if emb is not None:
        st.markdown("#### Map of 3,000 customer questions (MiniLM embeddings, t-SNE)")
        sel = alt.selection_point(fields=["category"], bind="legend")
        show_chart(alt.Chart(emb).mark_circle(size=22).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("category:N", legend=alt.Legend(orient="right")),
            opacity=alt.condition(sel, alt.value(.8), alt.value(.06)),
            tooltip=["category", "intent", "question"]).add_params(sel).properties(height=520))
        st.caption("Click a category in the legend to highlight it. Hover a point to read the question.")

    st.markdown("#### Nearest words in the LSA space")
    vocab, vecs = load_word_vectors()
    if vocab is not None:
        w = st.text_input("Word", value="refund")
        stem = textprep.preprocess(w.strip()) if w.strip() else ""
        hit = np.where(vocab == stem)[0]
        if len(hit):
            sims = vecs @ vecs[hit[0]]
            order = [i for i in np.argsort(-sims) if i != hit[0]][:10]
            show_df(pd.DataFrame({"word (stemmed)": vocab[order], "cosine similarity": sims[order].round(3)}))
        elif stem:
            st.info(f"`{stem}` isn't among the 4,000 most frequent words. Try order, cancel, password, invoice or delivery.")

    tk = rep.get("llama_tokens")
    if tk:
        st.markdown("#### How Llama sees the text (sub-word tokens)")
        c = st.columns(4)
        kpi(c[0], tk["question_tokens"]["mean"], "tokens per question (mean)")
        kpi(c[1], tk["answer_tokens"]["mean"], "tokens per answer (mean)")
        kpi(c[2], tk["tokens_per_word_answer"], "tokens per word")
        kpi(c[3], f"{tk['pct_answers_over_max_new_tokens']}%", f"answers longer than the {tk['max_new_tokens']}-token limit")
        ex = tk["slot_tokenisation_example"]
        st.markdown(f"`{ex['text']}` →  " + " ".join(f'<span class="chip">{esc(t)}</span>' for t in ex["tokens"]),
                    unsafe_allow_html=True)
        st.caption("A placeholder is split into several sub-word tokens. The model has to learn to produce the whole sequence.")


# ============================================================================
# 4 · Router
# ============================================================================
def page_router():
    st.markdown("## 4 · Intent router: Model 1 vs Model 2")
    if not need(router_cmp, "Router results"):
        return
    st.markdown("Both models read TF-IDF features of the cleaned question and predict one of the intents. "
                "The best model's prediction is passed to Llama as a routing note.")
    rc = router_cmp.copy()
    rc["model"] = rc.key.map(ROUTER_SHORT)
    show_df(rc[["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "cv_f1_macro_mean",
                "cv_f1_macro_std", "fit_seconds"]].round(4).rename(columns={
        "cv_f1_macro_mean": "5-fold CV macro-F1", "cv_f1_macro_std": "CV std", "fit_seconds": "fit time (s)"}))
    best = rc.sort_values("f1_macro", ascending=False).iloc[0]
    st.success(f"Best router: **{best.model}** (test macro-F1 {best.f1_macro:.4f}).")

    long = rc.melt(id_vars=["model"], value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    long["variable"] = long.variable.map({"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                                          "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)"})
    lo = max(0.0, math.floor((float(long.value.min()) - 0.02) * 50) / 50)
    show_chart(alt.Chart(long).mark_circle(size=220, opacity=1).encode(
        y=alt.Y("variable:N", title=None, sort=None, axis=YAX),
        x=alt.X("value:Q", scale=alt.Scale(domain=[lo, 1]), title=f"test-set score (axis starts at {lo:.2f})"),
        color=alt.Color("model:N", scale=ROUTER_COLORS), tooltip=["model", "variable", alt.Tooltip("value:Q", format=".4f")])
        .properties(height=220))

    pi = load_csv("router_per_intent.csv")
    if pi is not None:
        st.markdown("#### F1 per intent")
        pi["model"] = pi.key.map(ROUTER_SHORT)
        show_chart(alt.Chart(pi).mark_circle(size=90).encode(
            y=alt.Y("intent:N", sort=alt.EncodingSortField("f1", op="min"), title=None, axis=YAX),
            x=alt.X("f1:Q", scale=alt.Scale(domain=[max(0, pi.f1.min() - .05), 1])),
            color=alt.Color("model:N", scale=ROUTER_COLORS), tooltip=["intent", "model", alt.Tooltip("f1:Q", format=".3f")])
            .properties(height=600))

    cm = load_csv("router_confusion.csv")
    if cm is not None:
        st.markdown("#### Confusion matrix")
        c = st.columns([2, 2, 3])
        k = c[0].selectbox("Model", list(ROUTER_SHORT), format_func=ROUTER_SHORT.get, index=list(ROUTER_SHORT).index(best.key))
        off = c[1].toggle("Errors only", value=True)
        d = cm[cm.key == k]
        if off:
            d = d[d["true"] != d["pred"]]
        if len(d):
            show_chart(alt.Chart(d).mark_rect().encode(
                x=alt.X("pred:N", title="predicted"), y=alt.Y("true:N", title="true"),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="teals")), tooltip=["true", "pred", "count", alt.Tooltip("rate:Q", format=".1%")])
                .properties(height=560))
        else:
            st.success("No misclassifications on the test set.")

    tt = load_json("router_top_tokens.json")
    if tt:
        st.markdown("#### What the models learned (top tokens per intent)")
        intents = sorted(tt[best.key])
        it = st.selectbox("Intent", intents)
        c = st.columns(2)
        for col, k in zip(c, ["nb", "svm"]):
            if k in tt:
                col.markdown(f"**{ROUTER_SHORT[k]}**<br>" + " ".join(f'<span class="chip">{esc(t)}</span>' for t in tt[k][it]),
                             unsafe_allow_html=True)
    er = load_csv("router_errors.csv")
    if er is not None and len(er):
        with st.expander(f"Misclassified test questions ({best.model})"):
            show_df(er)

    st.markdown("#### Try the router")
    r = load_router(best.key)
    if r is not None:
        q = st.text_input("Customer message", value="i dont want my order anymore, how do i stop it?")
        if q.strip():
            top3, lab = router_top3(r, q)
            st.markdown(f"Cleaned text: `{textprep.router_text(q)}`")
            st.markdown(" ".join(f'<span class="chip">{esc(i)} · {s:.0%}</span>' for i, s in top3) + f' <span class="note">({lab})</span>',
                        unsafe_allow_html=True)


# ============================================================================
# 5 · Llama
# ============================================================================
def label_model(k):
    return {"base": "Base", "finetuned": "Fine-tuned", "reference": "Reference"}[k]


def page_llm():
    st.markdown("## 5 · Llama: base vs fine-tuned")
    if not need(summary, "Llama evaluation results"):
        return
    n = summary["n_test"]
    per = n // max(1, (data_summary or {}).get("n_intents", 27))
    st.markdown(f"Both models answered the same **{n} held-out test questions** ({per} per intent), "
                "with the same system prompt and routing note. Human-written Bitext answers are scored the same way "
                "as a benchmark wherever that is meaningful.")
    t1, t2, t3, t4, t5 = st.tabs(["Scorecard", "Training", "By intent", "Hallucinations", "Answer explorer"])

    with t1:
        rows = []
        for m in summary["metrics"]:
            sig = '<span class="badge b-ok">significant</span>' if m["significant"] else '<span class="badge b-ns">not significant</span>'
            win = label_model(m["better"])
            label = "Fluency (GPT-2 perplexity)" if m["key"] == "log_perplexity" else m["label"]
            if m["key"] == "log_perplexity":  # differences of log-perplexity shown as a ratio of perplexities
                diff = f"×{math.exp(m['diff_ft_minus_base']):.2f}<br><span class=\"note\">[×{math.exp(m['ci95'][0]):.2f}, ×{math.exp(m['ci95'][1]):.2f}]</span>"
            elif m["key"] in ("hallucination", "uses_placeholder"):
                diff = f"{100 * m['diff_ft_minus_base']:+.1f} pts<br><span class=\"note\">[{100 * m['ci95'][0]:+.1f}, {100 * m['ci95'][1]:+.1f}]</span>"
            else:
                diff = f"{m['diff_ft_minus_base']:+.3f}<br><span class=\"note\">[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]</span>"
            arrow = "↑ higher is better" if m["higher_is_better"] else "↓ lower is better"
            rows.append(f"""<tr><td><b>{esc(label)}</b><br><span class="note">{arrow}</span></td>
<td>{fmt_metric(m['key'], m['base'])}</td><td><b>{fmt_metric(m['key'], m['finetuned'])}</b></td>
<td>{fmt_metric(m['key'], m['reference'])}</td>
<td>{diff}</td>
<td>{m['p_value']:.2g}<br>{sig}</td><td><span class="badge {'b-ft' if m['better'] == 'finetuned' else 'b-base'}">{win}</span></td></tr>""")
        st.markdown(
            '<div class="card"><table style="width:100%;border-collapse:collapse;font-size:.88rem">'
            "<tr style='text-align:left;color:#64748B'><th>Metric</th><th>Base</th><th>Fine-tuned</th><th>Reference</th>"
            "<th>Δ (FT − base), 95% CI</th><th>p-value</th><th>Better</th></tr>" + "".join(rows) + "</table></div>",
            unsafe_allow_html=True)
        st.caption("Fluency is shown as perplexity under GPT-2 (geometric mean, lower is more fluent), and its difference as a "
                   "ratio (×0.80 means 20% lower perplexity). Differences are "
                   "tested with a paired bootstrap CI and Wilcoxon signed-rank (McNemar for yes/no metrics). The reference "
                   "column is blank for the correctness metrics, because a reference compared with itself is trivially perfect.")
        bar = []
        for m in summary["metrics"]:
            if m["key"] in ("log_perplexity",):
                continue
            for k in ("base", "finetuned", "reference"):
                if m[k] is not None:
                    bar.append({"metric": m["label"], "model": label_model(k), "value": m[k]})
        show_chart(alt.Chart(pd.DataFrame(bar)).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("metric:N", title=None, sort=None, axis=YAX), yOffset="model:N", x=alt.X("value:Q", title="mean score / rate"),
            color=alt.Color("model:N", scale=MODEL_COLORS), tooltip=["metric", "model", alt.Tooltip("value:Q", format=".3f")])
            .properties(height=440))
        lim = summary.get("pct_hit_token_limit", {})
        if lim:
            st.caption(f"Answers cut off at the token limit: base {100 * lim.get('base', 0):.1f}%, "
                       f"fine-tuned {100 * lim.get('finetuned', 0):.1f}%.")

    with t2:
        log = load_csv("train_log.csv")
        ti = (cfg or {}).get("train_info", {})
        c = st.columns(4)
        if ti:
            kpi(c[0], f"{ti.get('trainable_params', 0) / 1e6:.1f} M", "trainable LoRA parameters")
            kpi(c[1], f"{100 * ti.get('trainable_params', 0) / max(1, ti.get('total_params', 1)):.2f}%", "of all parameters")
            kpi(c[2], ti.get("optimizer_steps", "–"), "optimiser steps")
            kpi(c[3], f"{ti.get('train_minutes', '–')} min", f"training time on {(cfg or {}).get('gpu', 'GPU')}")
        if log is not None:
            log["smoothed"] = log.loss.rolling(8, min_periods=1).mean()
            base = alt.Chart(log).encode(x=alt.X("step:Q", title="optimiser step"))
            show_chart((base.mark_line(color="#CBD5E1").encode(y=alt.Y("loss:Q", title="training loss")) +
                        base.mark_line(color=C_FT, strokeWidth=3).encode(y="smoothed:Q")).properties(height=300))
        ppl = summary.get("reference_perplexity", {})
        if ppl:
            st.markdown(f"**Perplexity of the reference answers** (validation set): base **{ppl.get('base', float('nan')):.2f}** → "
                        f"fine-tuned **{ppl.get('finetuned', float('nan')):.2f}**. Lower means the model expects text like the target "
                        "support answers.")
        if cfg:
            st.caption(f"QLoRA: 4-bit NF4 base weights, LoRA r={cfg['lora']['r']}, α={cfg['lora']['alpha']}, dropout {cfg['lora']['dropout']}, "
                       f"lr {cfg['lr']}, {cfg['epochs']} epoch(s), effective batch {cfg['effective_batch']}, {cfg['n_train']:,} training examples.")

    with t3:
        pi = load_csv("llm_per_intent.csv")
        if pi is not None:
            mk = st.radio("Metric", ["bertscore", "rougeL", "hallucination"], horizontal=True,
                          format_func={"bertscore": "BERTScore", "rougeL": "ROUGE-L", "hallucination": "Hallucination rate"}.get)
            pi["model"] = pi.model.map(label_model)
            wide = pi.pivot_table(index="true_intent", columns="model", values=mk).reset_index()
            rule = alt.Chart(wide).mark_rule(color="#CBD5E1", strokeWidth=2).encode(
                y=alt.Y("true_intent:N", sort=alt.EncodingSortField("Fine-tuned", order="descending"), title=None, axis=YAX),
                x=alt.X("Base:Q", title=mk), x2="Fine-tuned:Q")
            pts = alt.Chart(pi).mark_circle(size=110, opacity=1).encode(
                y=alt.Y("true_intent:N", sort=alt.EncodingSortField("Fine-tuned", order="descending")), x=f"{mk}:Q",
                color=alt.Color("model:N", scale=MODEL_COLORS), tooltip=["true_intent", "model", alt.Tooltip(f"{mk}:Q", format=".3f")])
            show_chart((rule + pts).properties(height=640))

    with t4:
        h = metric(summary, "hallucination")
        c = st.columns(3)
        kpi(c[0], f"{100 * h['base']:.1f}%", "base answers with invented details")
        kpi(c[1], f"{100 * h['finetuned']:.1f}%", "fine-tuned answers with invented details", None, h["finetuned"] <= h["base"])
        kpi(c[2], f"{100 * (h['reference'] or 0):.1f}%", "reference answers flagged (detector check)")
        st.markdown(
            "An answer is flagged when it states a **specific detail the customer never gave**: an order or account number, a phone "
            "number, an e-mail, a link, a price or a time window (\"3–5 business days\"). A No-RAG bot can't know these. Writing a "
            "placeholder such as `{{Order Number}}` is the correct behaviour and isn't flagged. The reference answers are run through the "
            "same detector as a check on false alarms.")
        types = summary.get("hallucination_types", {})
        tl = [{"model": label_model(k), "type": t, "count": v} for k, d in types.items() for t, v in d.items()]
        if tl:
            show_chart(alt.Chart(pd.DataFrame(tl)).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("type:N", title=None, axis=YAX), yOffset="model:N", x=alt.X("count:Q", title="number of invented details"),
                color=alt.Color("model:N", scale=MODEL_COLORS), tooltip=["model", "type", "count"]).properties(height=260))
        if gens is not None:
            k = st.radio("Show flagged answers from", ["base", "finetuned"], horizontal=True, format_func=label_model)
            flagged = gens[gens[f"{k}_hallucination"] == 1]
            st.caption(f"{len(flagged)} flagged answers")
            for _, r in flagged.head(5).iterrows():
                ans = html_text(r[f"{k}_answer"])
                for vals in json.loads(r[f"{k}_invented"]).values():
                    for v in set(vals):
                        ans = ans.replace(esc(v), f"<mark>{esc(v)}</mark>")
                st.markdown(f'<div class="card"><h4>{esc(r.question)}</h4><div class="answer">{ans}</div></div>', unsafe_allow_html=True)

    with t5:
        if gens is None:
            need(None, "Generated answers")
            return
        c = st.columns([2, 3])
        intent = c[0].selectbox("Intent", ["(all)"] + sorted(gens.true_intent.unique()))
        g = gens if intent == "(all)" else gens[gens.true_intent == intent]
        order = c[1].selectbox("Sort by", ["Biggest gain from fine-tuning", "Biggest loss from fine-tuning", "Original order"])
        g = g.assign(gain=g.finetuned_bertscore - g.base_bertscore)
        if order.startswith("Biggest gain"):
            g = g.sort_values("gain", ascending=False)
        elif order.startswith("Biggest loss"):
            g = g.sort_values("gain")
        qi = st.selectbox("Question", range(len(g)), format_func=lambda i: g.iloc[i].question[:110])
        r = g.iloc[qi]
        st.markdown(f"**Customer:** {esc(r.question)}  \nTrue intent `{r.true_intent}` · router predicted `{r.predicted_intent}`")
        cols = st.columns(3)
        for col, k, cls in zip(cols, ["base", "finetuned", "reference"], ["b-base", "b-ft", "b-ref"]):
            text = r["reference"] if k == "reference" else r[f"{k}_answer"]
            chips = "" if k == "reference" else (
                f'<span class="chip">BERTScore {r[f"{k}_bertscore"]:.3f}</span><span class="chip">ROUGE-L {r[f"{k}_rougeL"]:.3f}</span>'
                f'<span class="chip">relevance {r[f"{k}_relevance"]:.3f}</span>'
                + ('<span class="badge b-warn">invented details</span>' if r[f"{k}_hallucination"] else ""))
            col.markdown(f'<div class="card"><span class="badge {cls}">{label_model(k)}</span><br>{chips}'
                         f'<div class="answer" style="margin-top:8px">{html_text(text)}</div></div>', unsafe_allow_html=True)


# ============================================================================
# 6 · Conclusions
# ============================================================================
def page_conclusions():
    st.markdown("## 6 · Conclusions and limitations")
    if summary:
        st.markdown("#### What the results show")
        for line in findings():
            st.markdown(f"- {line}")
    st.markdown(
        """
#### Interpretation
* **Routing is the easy part.** Bitext intents are lexically well separated, so a linear model on TF-IDF features
  routes almost perfectly. Most of the quality risk sits in generation.
* **Without retrieval, fine-tuning is how a model learns the domain.** The base model is fluent, but it writes like a
  general assistant. The fine-tuned model adopts the company's answer structure and placeholder convention, which
  shows up in correctness, reference perplexity and placeholder use.
* **Grounding by placeholder.** A No-RAG bot can't know an order's status. The safest behaviour is to ask for, or
  refer to, the detail with a placeholder rather than invent it. The hallucination check measures exactly this.

#### Limitations
* Helpfulness is a rule-based proxy (concrete actions and steps), not a human judgement.
* The hallucination detector is rule-based. It catches invented numbers, links, prices and time windows, but not
  invented policies phrased in words.
* Bitext is a synthetic dataset, so real customer messages will be messier.
* One training run with one seed, on a test sample of a few questions per intent.
* The hosted demo replays recorded answers. Llama itself runs in Colab, not on free Streamlit hosting.
        """
    )


ROUTES = {"Overview": page_overview, "Chat demo": page_chat, "1 · Data cleaning": page_data, "2 · Exploration": page_eda,
          "3 · Text representation": page_repr, "4 · Intent router": page_router,
          "5 · Llama: base vs fine-tuned": page_llm, "6 · Conclusions": page_conclusions}
ROUTES[page]()
