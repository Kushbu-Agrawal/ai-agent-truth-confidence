"""
Can an AI Agent Tell When It's Being Told the Truth?
A Study of Confidence Bias in Agentic AI Decision-Making (CA3)
Streamlit app version of the Colab notebook.

Authors: Kushbu Agrawal (CSE, PRN 23070122267) and Sonalika Singha (CSE, PRN 23070122207)
"""
import re
import json
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss, confusion_matrix
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page + global style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Agent: Truth vs Confidence", page_icon="🧪", layout="wide")

plt.rcParams.update({"text.color": "black", "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black", "axes.edgecolor": "black", "font.family": "serif",
                     "font.size": 10, "figure.dpi": 110, "savefig.bbox": "tight"})
BLUE = "#7fa6c9"; GREY = "#b8b8b8"; ORANGE = "#e3a25b"; GREEN = "#8fc08f"
TONES = ["assertive", "neutral", "hedged"]
AGENTS = ["baseline", "neutralized", "balanced"]
AGENT_NAMES = {"baseline": "Baseline", "neutralized": "Cue-neutralised", "balanced": "Balanced-trained"}
AGENT_COLORS = {"baseline": ORANGE, "neutralized": GREEN, "balanced": BLUE}
EVID_P = {"claims": 0.50, "reviews": 0.40, "iot": 0.70}   # chance that a supporting Record exists
TRAIN_FRACTION = 0.70
SWEEP_P = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CMAP = LinearSegmentedColormap.from_list("lb", ["#ffffff", "#7fa6c9"])

# ---------------------------------------------------------------------------
# 1. Confidence vocabulary and Hindi phrase table
# ---------------------------------------------------------------------------
A_PRE = ["Definitely true:", "Absolutely certain:", "100% confirmed:", "Without a doubt,",
         "It is undeniably proven that", "Everyone knows that", "Beyond question,", "Guaranteed fact:"]
A_SUF = ["(100% verified).", "no question about it.", "I am absolutely sure."]
H_PRE = ["Maybe", "I think", "Possibly,", "Reportedly,", "Some say", "It seems that", "Apparently,", "Not sure, but"]
H_SUF = ["but I am not sure.", "though this is unverified.", "or so I heard, could be wrong.", "if I recall correctly, maybe."]
WRAP = {"assertive": {"pre": A_PRE, "suf": A_SUF}, "hedged": {"pre": H_PRE, "suf": H_SUF}}
GEN_A = ["definitely", "absolutely", "certainly", "undeniably", "guaranteed", "confirmed", "proven",
         "obviously", "for sure", "100%", "no doubt"]
GEN_H = ["maybe", "perhaps", "possibly", "probably", "reportedly", "allegedly", "unverified",
         "apparently", "i think", "not sure", "i guess", "might"]

HI_TABLE = {
    "Definitely true:": "बिल्कुल सच:", "Absolutely certain:": "पूरी तरह पक्का:", "100% confirmed:": "100% पुष्ट:",
    "Without a doubt,": "बिना किसी शक के,", "It is undeniably proven that": "यह निर्विवाद रूप से सिद्ध है कि",
    "Everyone knows that": "सबको पता है कि", "Beyond question,": "इसमें कोई सवाल ही नहीं,",
    "Guaranteed fact:": "गारंटीशुदा तथ्य:", "(100% verified).": "(100% सत्यापित)।",
    "no question about it.": "इसमें कोई शक नहीं।", "I am absolutely sure.": "मुझे पूरा यकीन है।",
    "Maybe": "शायद", "I think": "मुझे लगता है", "Possibly,": "संभवतः,", "Reportedly,": "कथित तौर पर,",
    "Some say": "कुछ लोग कहते हैं", "It seems that": "ऐसा लगता है कि", "Apparently,": "ऐसा प्रतीत होता है,",
    "Not sure, but": "पक्का नहीं, लेकिन", "but I am not sure.": "लेकिन मुझे पक्का नहीं पता।",
    "though this is unverified.": "हालांकि यह अपुष्ट है।", "or so I heard, could be wrong.": "ऐसा मैंने सुना है, गलत भी हो सकता है।",
    "if I recall correctly, maybe.": "अगर मुझे ठीक याद है, शायद।"}

LEX = []   # (phrase, class, language)
for p in A_PRE + A_SUF + GEN_A: LEX.append((p, "A", "en"))
for p in H_PRE + H_SUF + GEN_H: LEX.append((p, "H", "en"))
for p in A_PRE + A_SUF: LEX.append((HI_TABLE[p], "A", "hi"))
for p in H_PRE + H_SUF: LEX.append((HI_TABLE[p], "H", "hi"))

_RX = {}
def get_rx(langs):
    langs = tuple(langs)
    if langs not in _RX:
        items = sorted({(p.lower(), c) for p, c, l in LEX if l in langs}, key=lambda x: -len(x[0]))
        pat = "|".join(re.escape(p) for p, _ in items)
        _RX[langs] = (re.compile(r"(?<!\w)(?:%s)(?!\w)" % pat, re.I), dict(items))
    return _RX[langs]

def strip_cues(text, langs=("en",)):
    """Split a sentence into (neutral core, list of (phrase, class))."""
    rx, cmap = get_rx(langs)
    found = []
    def rep(m):
        found.append((m.group(0).lower(), cmap[m.group(0).lower()])); return " "
    neutral = re.sub(r"\s+", " ", rx.sub(rep, text)).strip(" ,:;")
    return neutral, found

def tok(s): return re.findall(r"[a-z0-9]+", s.lower())

# ---------------------------------------------------------------------------
# 2. Dataset generator
# ---------------------------------------------------------------------------
CAP = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Egypt": "Cairo", "Kenya": "Nairobi", "Canada": "Ottawa",
       "Brazil": "Brasilia", "Australia": "Canberra", "India": "New Delhi", "China": "Beijing", "Russia": "Moscow",
       "Spain": "Madrid", "Germany": "Berlin", "Turkey": "Ankara", "Argentina": "Buenos Aires", "Nigeria": "Abuja",
       "Thailand": "Bangkok", "Norway": "Oslo", "Peru": "Lima", "Greece": "Athens", "Poland": "Warsaw",
       "Portugal": "Lisbon", "Austria": "Vienna", "Sweden": "Stockholm", "Cuba": "Havana", "Chile": "Santiago",
       "Ghana": "Accra", "Nepal": "Kathmandu", "Iran": "Tehran", "Ireland": "Dublin"}
ELEM = {"Hydrogen": "H", "Helium": "He", "Lithium": "Li", "Carbon": "C", "Nitrogen": "N", "Oxygen": "O", "Sodium": "Na",
        "Magnesium": "Mg", "Aluminium": "Al", "Silicon": "Si", "Phosphorus": "P", "Sulfur": "S", "Chlorine": "Cl",
        "Potassium": "K", "Calcium": "Ca", "Iron": "Fe", "Copper": "Cu", "Zinc": "Zn", "Silver": "Ag", "Gold": "Au"}
TOPICS = ["Unemployment", "Inflation", "Fuel prices", "Retail sales", "Rainfall", "Vegetable prices"]
REGIONS = ["Nashik", "Pune", "Mumbai", "Nagpur", "Kolhapur", "Aurangabad", "Thane", "Solapur"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
PRODUCTS = ["phone", "laptop", "tablet", "smartwatch", "speaker", "earbuds", "camera", "power bank"]
ZONES = ["Zone A", "Zone B", "Zone C", "Zone D", "Boiler Room", "Cold Store"]

def num_pair(rng, lo, hi, spread):
    v = rng.randint(lo, hi)
    w = v
    while w == v or w < lo - spread: w = v + rng.choice([-1, 1]) * rng.randint(1, spread)
    return str(v), str(w)

def sample_group(rng, domain, gid):
    if domain == "claims":
        t = rng.random()
        if t < 0.4:
            c = rng.choice(list(CAP)); w = rng.choice([x for x in CAP.values() if x != CAP[c]])
            tr, fa = f"{CAP[c]} is the capital of {c}.", f"{w} is the capital of {c}."
        elif t < 0.6:
            e = rng.choice(list(ELEM)); w = rng.choice([x for x in ELEM.values() if x != ELEM[e]])
            tr, fa = f"The chemical symbol for {e} is {ELEM[e]}.", f"The chemical symbol for {e} is {w}."
        else:
            tp, rg, mo, vb = rng.choice(TOPICS), rng.choice(REGIONS), rng.choice(MONTHS), rng.choice(["rose by", "fell by"])
            a, b = num_pair(rng, 2, 18, 4)
            tr, fa = (f"{tp} in {rg} {vb} {a} percent in {mo}.", f"{tp} in {rg} {vb} {b} percent in {mo}.")
    elif domain == "reviews":
        p = rng.choice(PRODUCTS); k = rng.random()
        if k < 0.4:
            a, b = num_pair(rng, 6, 40, 6); tr, fa = f"The {p} battery lasts {a} hours.", f"The {p} battery lasts {b} hours."
        elif k < 0.7:
            a, b = num_pair(rng, 100, 900, 90); tr, fa = f"The {p} weighs {a} grams.", f"The {p} weighs {b} grams."
        else:
            a, b = num_pair(rng, 1, 5, 2); tr, fa = f"This {p} has a {a} year warranty.", f"This {p} has a {b} year warranty."
    else:
        k = rng.random()
        if k < 0.4:
            s, z = rng.randint(10, 99), rng.choice(ZONES); a, b = num_pair(rng, 30, 110, 15)
            tr, fa = (f"Sensor S{s} in {z} reported {a} degrees Celsius.", f"Sensor S{s} in {z} reported {b} degrees Celsius.")
        elif k < 0.7:
            s = rng.randint(1, 40); a, b = num_pair(rng, 20, 90, 12)
            tr, fa = f"Pump P{s} pressure reached {a} psi.", f"Pump P{s} pressure reached {b} psi."
        else:
            s = rng.randint(1, 12); a, b = num_pair(rng, 2, 60, 8)
            tr, fa = f"Gate G{s} opened {a} times today.", f"Gate G{s} opened {b} times today."
    return {"gid": gid, "domain": domain, "true": tr, "false": fa}

def make_groups(rng, n_groups):
    gs, gid = [], 0
    for d in EVID_P:
        for _ in range(n_groups):
            gs.append(sample_group(rng, d, gid)); gid += 1
    rng.shuffle(gs); n = int(len(gs) * TRAIN_FRACTION)
    return gs[:n], gs[n:]

TRAIN_TONE = {True: {"assertive": .50, "neutral": .35, "hedged": .15}, False: {"assertive": .15, "neutral": .35, "hedged": .50}}
BAL_TONE = {True: {t: 1 / 3 for t in TONES}, False: {t: 1 / 3 for t in TONES}}

def mk_item(rng, g, label, tone, evid_p, lang="en"):
    pre = suf = None
    if tone != "neutral":
        if rng.random() < 0.7: pre = rng.choice(WRAP[tone]["pre"])
        else: suf = rng.choice(WRAP[tone]["suf"])
    rec = ("Record: " + g["true"]) if rng.random() < evid_p else None
    return {"gid": g["gid"], "domain": g["domain"], "label": label, "tone": tone,
            "core": g["true"] if label else g["false"], "pre": pre, "suf": suf, "record": rec, "lang": lang}

DECAP = {"The", "This", "Sensor", "Pump", "Gate", "Unemployment", "Inflation", "Fuel", "Retail", "Rainfall", "Vegetable"}
def decap(c): return c[0].lower() + c[1:] if c.split(" ", 1)[0] in DECAP else c
def tr_phrase(p, lang): return p if lang == "en" else HI_TABLE[p]
def render(it):
    c, L = it["core"], it["lang"]
    if isinstance(it["pre"], str): return f"{tr_phrase(it['pre'], L)} {decap(c)}"
    if isinstance(it["suf"], str):
        s = tr_phrase(it["suf"], L); return c.rstrip(".") + (" " if s.startswith("(") else ", ") + s
    return c

def build_train(groups, rng, dist, k=2):
    out = []
    for g in groups:
        for lab in (True, False):
            for _ in range(k):
                d = dist[lab]; t = rng.choices(TONES, weights=[d[x] for x in TONES])[0]
                out.append(mk_item(rng, g, lab, t, EVID_P[g["domain"]]))
    return out

def build_test(groups, rng, p_override=None):
    return [mk_item(rng, g, lab, t, EVID_P[g["domain"]] if p_override is None else p_override)
            for g in groups for lab in (True, False) for t in TONES]

def as_lang(items, lang): return [dict(i, lang=lang) for i in items]

# ---------------------------------------------------------------------------
# 3. Agent (feed-forward neural network) + metrics
# ---------------------------------------------------------------------------
def tone_tokens(s): return [w.strip(".,:;()!") for w in s.split() if w.strip(".,:;()!")]

class Agent:
    """Neural fact-verification agent. use_tone=False gives the cue-neutralised (debiased) agent."""
    def __init__(self, seed, langs=("en",), use_tone=True):
        self.seed, self.langs, self.use_tone = seed, langs, use_tone
        self.vec = TfidfVectorizer(analyzer=tone_tokens); self.sc = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, max_iter=500,
                                 early_stopping=True, validation_fraction=0.15, n_iter_no_change=15, random_state=seed)
    def feats(self, items):
        num, tone_txt = [], []
        for it in items:
            neutral, ph = strip_cues(render(it), self.langs)
            ct = set(tok(neutral)); rec = it["record"] if isinstance(it["record"], str) else None
            if rec:
                rt = set(tok(rec.replace("Record:", "")))
                mism, extra = len(ct - rt), len(rt - ct)
                jac = len(ct & rt) / max(1, len(ct | rt)); has = 1
            else: mism = extra = 0; jac = 0.0; has = 0
            nA = sum(1 for _, c in ph if c == "A"); nH = sum(1 for _, c in ph if c == "H")
            row = [has, mism, extra, jac, float(has and mism == 0)]
            if self.use_tone: row += [nA, nH, nA - nH, len(" ".join(p for p, _ in ph).split())]
            num.append(row); tone_txt.append(" ".join(p for p, _ in ph))
        return np.array(num, float), tone_txt
    def _X(self, items, fit):
        num, tt = self.feats(items)
        num = self.sc.fit_transform(num) if fit else self.sc.transform(num)
        if not self.use_tone: return csr_matrix(num)
        T = self.vec.fit_transform(tt) if fit else self.vec.transform(tt)
        return hstack([csr_matrix(num), T]).tocsr()
    def fit(self, items):
        self.clf.fit(self._X(items, True), np.array([i["label"] for i in items], int)); return self
    def proba(self, items): return self.clf.predict_proba(self._X(items, False))[:, 1]

def frame(items, p):
    df = pd.DataFrame({"domain": [i["domain"] for i in items], "label": [i["label"] for i in items],
                       "tone": [i["tone"] for i in items], "evid": [i["record"] is not None for i in items]})
    df["p"] = p; df["pred"] = df.p > 0.5; df["correct"] = df.pred == df.label
    df["conf"] = np.maximum(df.p, 1 - df.p); return df

def ece(df, bins=10):
    e, n = 0.0, len(df); edges = np.linspace(0.5, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (df.conf > lo) & (df.conf <= hi) if lo > 0.5 else (df.conf >= lo) & (df.conf <= hi)
        if m.sum(): e += m.sum() / n * abs(df.conf[m].mean() - df.correct[m].mean())
    return e

def cbi(df):
    a = df.groupby(["label", "tone"]).correct.mean()
    return (a[(True, "assertive")] + a[(False, "hedged")]) / 2 - (a[(False, "assertive")] + a[(True, "hedged")]) / 2

def summarise(df):
    pr, rc, f1, _ = precision_recall_fscore_support(df.label, df.pred, average="macro", zero_division=0)
    return {"acc": df.correct.mean(), "prec": pr, "rec": rc, "f1": f1, "cbi": cbi(df), "ece": ece(df),
            "brier": brier_score_loss(df.label.astype(int), df.p),
            "far": 1 - df[(~df.label) & (df.tone == "assertive")].correct.mean(),
            "trr": 1 - df[(df.label) & (df.tone == "hedged")].correct.mean(),
            "acc_noevid": df[~df.evid].correct.mean(), "acc_evid": df[df.evid].correct.mean()}

# ---------------------------------------------------------------------------
# 4. Experiment
# ---------------------------------------------------------------------------
NOISE = ["Further details will be shared in the next update.", "The team reviewed the situation during the weekly meeting.",
         "Please refer to the attached summary for context.", "This message was generated automatically at 09:14.",
         "Contact the operations desk for any questions.", "Similar items were logged earlier this month.",
         "An archived copy is stored in the shared folder.", "Reviewers will follow up if more information is needed.",
         "The report is circulated to all registered users.", "Updates are published every morning."]

def jacc(a, b): a, b = set(a), set(b); return len(a & b) / max(1, len(a | b))
def rouge1(hyp, ref):
    h, r = tok(hyp), tok(ref)
    if not h or not r: return 0.0
    ov = sum((Counter(h) & Counter(r)).values()); p, rc = ov / len(h), ov / len(r)
    return 0 if ov == 0 else 2 * p * rc / (p + rc)

def make_doc(rng, it):
    noise = rng.sample(NOISE, 6)
    if it["record"]: noise.insert(rng.randint(0, len(noise)), it["record"])
    return [render(it)] + noise

def summarise_doc(sents, mode):
    if mode == "full": return sents
    if mode == "lead2": return sents[:2]
    q = set(tok(strip_cues(sents[0])[0]))
    rest = sorted(sents[1:], key=lambda s: -jacc(q, tok(s.replace("Record:", ""))))
    return [sents[0], rest[0]]

def run_experiment(seeds, n_groups, do_mt=True, do_sum=True, progress=None):
    res = {a: [] for a in AGENTS}; frames = {a: [] for a in AGENTS}; cells = {a: [] for a in AGENTS}
    sweep = {a: {} for a in AGENTS}
    mt_res = {k: [] for k in ["en_only", "mt_aug", "mt_aug_neut", "mt_aug_on_en"]}
    sum_res = {k: [] for k in ["full", "lead2", "query"]}
    neut_recov = {"en": [], "hi": []}
    train_stats = []; ALL = {}; demo_agents = None
    for i, seed in enumerate(seeds):
        if progress: progress.progress(i / len(seeds), text=f"Seed {seed}  ({i + 1}/{len(seeds)}): training agents and evaluating…")
        rng = random.Random(seed); np.random.seed(seed)
        trg, teg = make_groups(rng, n_groups)
        tr_b = build_train(trg, rng, TRAIN_TONE); tr_bal = build_train(trg, rng, BAL_TONE); te = build_test(teg, rng)
        ag = {"baseline": Agent(seed).fit(tr_b), "neutralized": Agent(seed, use_tone=False).fit(tr_b),
              "balanced": Agent(seed).fit(tr_bal)}
        for a in AGENTS:
            df = frame(te, ag[a].proba(te)); frames[a].append(df); res[a].append(summarise(df))
            cells[a].append(df.groupby(["label", "tone"]).correct.mean())
        if i == 0:
            ALL["train0"] = pd.DataFrame(tr_b); ALL["test0"] = pd.DataFrame(te); demo_agents = ag
        trdf = pd.DataFrame(tr_b)
        train_stats.append(trdf.groupby(["label", "tone"]).size().unstack() / trdf.groupby("label").size().values[:, None])
        # evidence sweep
        for pe in SWEEP_P:
            tes = build_test(teg, random.Random(seed * 100 + int(pe * 10)), p_override=pe)
            for a in AGENTS:
                s = summarise(frame(tes, ag[a].proba(tes)))
                sweep[a].setdefault(pe, []).append({"far": s["far"], "trr": s["trr"], "acc": s["acc"], "cbi": s["cbi"]})
        # neutraliser recovery
        for lang, langs in (("en", ("en",)), ("hi", ("en", "hi"))):
            ti = as_lang(te, lang); ok = sum(tok(strip_cues(render(x), langs)[0]) == tok(x["core"]) for x in ti)
            neut_recov[lang].append(ok / len(ti))
        # machine-translation experiment
        if do_mt:
            te_hi = as_lang(te, "hi"); tr_hi = as_lang(tr_b, "hi")
            a_mt = Agent(seed, langs=("en", "hi")).fit(tr_b + tr_hi)
            a_mtn = Agent(seed, langs=("en", "hi"), use_tone=False).fit(tr_b + tr_hi)
            mt_res["en_only"].append(summarise(frame(te_hi, ag["baseline"].proba(te_hi))))
            mt_res["mt_aug"].append(summarise(frame(te_hi, a_mt.proba(te_hi))))
            mt_res["mt_aug_neut"].append(summarise(frame(te_hi, a_mtn.proba(te_hi))))
            mt_res["mt_aug_on_en"].append(summarise(frame(te, a_mt.proba(te))))
        # summarisation experiment
        if do_sum:
            docs = [make_doc(rng, x) for x in te]
            for mode in ("full", "lead2", "query"):
                mod, kr, rg, cp = [], [], [], []
                for it, d in zip(te, docs):
                    s = summarise_doc(d, mode); has_rec = any(x.startswith("Record:") for x in s)
                    mod.append(dict(it, record=it["record"] if has_rec else None))
                    if it["record"]: kr.append(has_rec)
                    ref = render(it) + " " + (it["record"] or "")
                    rg.append(rouge1(" ".join(s), ref)); cp.append(len(" ".join(s).split()) / len(" ".join(d).split()))
                s_ = summarise(frame(mod, ag["baseline"].proba(mod)))
                s_.update({"key_recall": float(np.mean(kr)), "rouge1": float(np.mean(rg)), "compression": float(np.mean(cp))})
                sum_res[mode].append(s_)
    if progress: progress.progress(1.0, text="Aggregating results and running statistical tests…")
    return {"res": res, "frames": frames, "cells": cells, "sweep": sweep, "mt_res": mt_res if do_mt else None,
            "sum_res": sum_res if do_sum else None, "neut_recov": neut_recov, "train_stats": train_stats,
            "ALL": ALL, "n_groups": n_groups, "seeds": list(seeds), "demo_agents": demo_agents}

def agg(lst):
    keys = lst[0].keys()
    return {k: (float(np.mean([r[k] for r in lst])), float(np.std([r[k] for r in lst], ddof=1))) for k in keys}

def aggregate(raw):
    res, frames, cells, sweep = raw["res"], raw["frames"], raw["cells"], raw["sweep"]
    R = {"main": {a: agg(res[a]) for a in AGENTS}, "recov": {k: float(np.mean(v)) for k, v in raw["neut_recov"].items()}}
    R["mt"] = {k: agg(v) for k, v in raw["mt_res"].items()} if raw["mt_res"] else None
    R["sum"] = {k: agg(v) for k, v in raw["sum_res"].items()} if raw["sum_res"] else None
    R["cells"] = {a: {f"{l}|{t}": [float(np.mean([c[(l, t)] for c in cells[a]])), float(np.std([c[(l, t)] for c in cells[a]], ddof=1))]
                      for l in (True, False) for t in TONES} for a in AGENTS}
    R["tone_train"] = {f"{l}|{t}": float(np.mean([s.loc[l, t] for s in raw["train_stats"]])) for l in (True, False) for t in TONES}
    pool = {a: pd.concat(frames[a], ignore_index=True) for a in AGENTS}
    R["domain"] = {a: {d: [float(np.mean([cbi(f[f.domain == d]) for f in frames[a]])),
                           float(np.std([cbi(f[f.domain == d]) for f in frames[a]], ddof=1))] for d in EVID_P} for a in AGENTS}
    R["sweep"] = {a: {str(p): {k: float(np.mean([r[k] for r in v])) for k in ("far", "trr", "acc", "cbi")}
                      for p, v in sweep[a].items()} for a in AGENTS}
    # statistical tests
    fb = pool["baseline"]; fneg = fb[~fb.label]
    ct = pd.crosstab(fneg.tone, fneg.correct); chi2, pval, dof, _ = stats.chi2_contingency(ct)
    T = {"chi2": float(chi2), "p": float(pval), "dof": int(dof),
         "cramers_v": float(np.sqrt(chi2 / (len(fneg) * (min(ct.shape) - 1)))), "n_false": int(len(fneg))}
    cb = [r["cbi"] for r in res["baseline"]]; cn = [r["cbi"] for r in res["neutralized"]]; cbal = [r["cbi"] for r in res["balanced"]]
    try:
        t1 = stats.ttest_rel(cb, cn); t2 = stats.ttest_1samp(cb, 0); t3 = stats.ttest_rel(cb, cbal)
        T.update({"t_base_vs_neut": [float(t1.statistic), float(t1.pvalue)], "t_base_gt0": [float(t2.statistic), float(t2.pvalue)],
                  "t_base_vs_bal": [float(t3.statistic), float(t3.pvalue)]})
    except Exception:
        pass
    try: T["wilcoxon_p"] = float(stats.wilcoxon(cb, cn).pvalue)
    except Exception: T["wilcoxon_p"] = None
    T["fa_acc_assertive"] = float(fneg[fneg.tone == "assertive"].correct.mean())
    T["fa_acc_hedged"] = float(fneg[fneg.tone == "hedged"].correct.mean())
    for key, sub in (("noevid", fneg[~fneg.evid]), ("evid", fneg[fneg.evid])):
        try:
            c2, p2, _, _ = stats.chi2_contingency(pd.crosstab(sub.tone, sub.correct))
            T[f"chi2_{key}"], T[f"p_{key}"] = float(c2), float(p2)
        except Exception:
            T[f"chi2_{key}"], T[f"p_{key}"] = None, None
    R["tests"] = T
    R["calib_tone"] = {t: {"conf": float(fb[fb.tone == t].conf.mean()), "acc": float(fb[fb.tone == t].correct.mean())} for t in TONES}
    R["cm"] = {a: confusion_matrix(pool[a].label, pool[a].pred, labels=[False, True]).tolist() for a in AGENTS}
    R["n"] = {"train": len(raw["ALL"]["train0"]), "test": len(raw["ALL"]["test0"]),
              "groups": raw["n_groups"] * 3, "seeds": len(raw["seeds"])}
    return R, pool

# ---------------------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------------------
def fig_dist(R):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    train = {lab: {t: R["tone_train"][f"{lab}|{t}"] for t in TONES} for lab in (True, False)}
    for ax, (nm, dist) in zip(axs, (("Training set (biased)", train), ("Test set (balanced)", BAL_TONE))):
        bottom = np.zeros(2)
        for t, c in zip(TONES, (ORANGE, GREY, GREEN)):
            v = np.array([dist[True][t], dist[False][t]])
            ax.bar(["True claims", "False claims"], v, bottom=bottom, color=c, ec="black", lw=0.8, label=t)
            for i, x in enumerate(v): ax.text(i, bottom[i] + x / 2, f"{x:.0%}", ha="center", va="center", fontsize=8)
            bottom += v
        ax.set_title(nm, fontsize=10)
    axs[0].set_ylabel("Share of items"); axs[1].legend(loc="upper right", fontsize=8, frameon=True)
    return fig

def fig_heatmap(R):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, a in zip(axs, AGENTS):
        M = np.array([[R["cells"][a][f"{l}|{t}"][0] for t in TONES] for l in (True, False)])
        sns.heatmap(M, annot=True, fmt=".2f", cmap=CMAP, vmin=0.3, vmax=1, cbar=False, ax=ax, xticklabels=TONES,
                    yticklabels=["True claim", "False claim"], annot_kws={"color": "black", "fontsize": 10},
                    linewidths=1, linecolor="black")
        ax.set_title(f"{AGENT_NAMES[a]} agent", fontsize=10)
    return fig

def fig_bias(R):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (k, nm) in zip(axs, (("far", "False acceptance (false + assertive)"), ("trr", "True rejection (true + hedged)"), ("cbi", "Confidence Bias Index"))):
        m = [R["main"][a][k][0] for a in AGENTS]; s = [R["main"][a][k][1] for a in AGENTS]
        b = ax.bar(["Baseline", "Neutralised", "Balanced"], m, yerr=s, capsize=4, color=[ORANGE, GREEN, BLUE], ec="black", lw=0.8)
        for r, v in zip(b, m): ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(nm, fontsize=9.5); ax.set_ylim(min(0, min(m) - 0.05), max(m) * 1.3 + 0.05)
    return fig

def fig_calib(R, pool):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, a in zip(axs, ("baseline", "neutralized")):
        df = pool[a]; edges = np.linspace(0.5, 1, 6); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (df.conf > lo) & (df.conf <= hi) if lo > .5 else (df.conf >= lo) & (df.conf <= hi)
            if m.sum() > 20: xs.append(df.conf[m].mean()); ys.append(df.correct[m].mean())
        ax.plot([0.5, 1], [0.5, 1], "--", c="black", lw=1, label="Perfect calibration")
        ax.plot(xs, ys, "o-", c="#2d6a9f", lw=1.6, label="Agent")
        ax.set_xlabel("Stated confidence"); ax.set_ylabel("Observed accuracy")
        ax.set_title(f"{AGENT_NAMES[a]} (ECE = {R['main'][a]['ece'][0]:.3f})", fontsize=10); ax.legend(fontsize=8); ax.set_ylim(0.4, 1.02)
    return fig

def fig_domain(R):
    fig, ax = plt.subplots(figsize=(7, 3.4)); w = 0.26
    for i, a in enumerate(AGENTS):
        v = [R["domain"][a][d][0] for d in EVID_P]; e = [R["domain"][a][d][1] for d in EVID_P]
        ax.bar(np.arange(3) + (i - 1) * w, v, w, yerr=e, capsize=3, color=AGENT_COLORS[a], ec="black", lw=0.8, label=AGENT_NAMES[a])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Claims (news/facts)", "Reviews", "IoT alerts"])
    ax.set_ylabel("Confidence Bias Index"); ax.legend(fontsize=8)
    return fig

def fig_sweep(R):
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a in AGENTS:
        axs[0].plot(SWEEP_P, [R["sweep"][a][str(p)]["far"] for p in SWEEP_P], "o-", c=AGENT_COLORS[a], mec="black", label=AGENT_NAMES[a])
        axs[1].plot(SWEEP_P, [R["sweep"][a][str(p)]["acc"] for p in SWEEP_P], "o-", c=AGENT_COLORS[a], mec="black", label=AGENT_NAMES[a])
    axs[0].set_ylabel("False acceptance rate"); axs[1].set_ylabel("Overall accuracy")
    for ax in axs: ax.set_xlabel("Probability that a supporting Record exists"); ax.legend(fontsize=8)
    return fig

def fig_mt(R):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3)); names = ["en_only", "mt_aug", "mt_aug_neut"]
    lab = ["English-only\nagent", "MT-augmented\nagent", "MT-augmented\n+ neutraliser"]
    for ax, k, nm in zip(axs, ("acc", "cbi"), ("Accuracy on Hindi test", "Confidence Bias Index on Hindi test")):
        m = [R["mt"][n][k][0] for n in names]; s = [R["mt"][n][k][1] for n in names]
        b = ax.bar(lab, m, yerr=s, capsize=4, color=[ORANGE, BLUE, GREEN], ec="black", lw=0.8)
        for r, v in zip(b, m): ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(nm, fontsize=10); ax.tick_params(axis="x", labelsize=8); ax.set_ylim(0, 1.05 if k == "acc" else max(m) * 1.35 + .05)
    return fig

def fig_sum(R):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.3)); names = ["full", "lead2", "query"]; lab = ["Full\nreport", "Lead-2", "Query-\nfocused"]
    for ax, k, nm in zip(axs, ("key_recall", "acc", "far"), ("Record retained (key recall)", "Agent accuracy", "False acceptance rate")):
        m = [R["sum"][n][k][0] for n in names]; s = [R["sum"][n][k][1] for n in names]
        b = ax.bar(lab, m, yerr=s, capsize=4, color=[GREY, ORANGE, GREEN], ec="black", lw=0.8)
        for r, v in zip(b, m): ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_title(nm, fontsize=10); ax.set_ylim(0, 1.1)
    return fig

def fig_cm(R):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
    for ax, a in zip(axs, AGENTS):
        sns.heatmap(np.array(R["cm"][a]), annot=True, fmt="d", cmap=CMAP, cbar=False, ax=ax,
                    xticklabels=["Pred false", "Pred true"], yticklabels=["Actual false", "Actual true"],
                    annot_kws={"color": "black"}, linewidths=1, linecolor="black")
        ax.set_title(AGENT_NAMES[a], fontsize=10)
    return fig

def fig_gap(R):
    fig, ax = plt.subplots(figsize=(6.5, 3.3)); w = .35
    ax.bar(np.arange(3) - w / 2, [R["calib_tone"][t]["conf"] for t in TONES], w, color=ORANGE, ec="black", lw=.8, label="Mean stated confidence")
    ax.bar(np.arange(3) + w / 2, [R["calib_tone"][t]["acc"] for t in TONES], w, color=BLUE, ec="black", lw=.8, label="Actual accuracy")
    ax.set_xticks(range(3)); ax.set_xticklabels(TONES); ax.set_ylim(0.4, 1.05); ax.legend(fontsize=8, loc="lower center")
    return fig

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis("off")
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="black", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    box(0.3, 6.6, 4.6, 2.4, "Source claims\n(facts, reviews,\nIoT alert logs)", BLUE)
    box(6.2, 6.6, 4.6, 2.4, "Confidence variant\ngenerator\n(assertive, neutral,\nhedged)", ORANGE)
    box(12.1, 6.6, 4.0, 2.4, "Pre-processing\n(cue split,\ntokenise)", GREY)
    box(17.4, 6.6, 4.3, 2.4, "Feature engineering\n(consistency + tone\nfeatures)", GREY)
    box(0.3, 1.0, 4.6, 2.4, "Agent (feed-forward\nneural network)\ntrust / reject", GREEN)
    box(6.2, 1.0, 4.6, 2.4, "Decision +\nstated confidence", GREEN)
    box(12.1, 1.0, 4.0, 2.4, "Evaluation\n(accuracy, CBI, ECE,\nBrier, F1)", BLUE)
    arrow(4.9, 7.8, 6.2, 7.8); arrow(10.8, 7.8, 12.1, 7.8); arrow(16.1, 7.8, 17.4, 7.8)
    arrow(19.5, 6.6, 19.5, 5.0); arrow(19.5, 5.0, 2.6, 5.0); arrow(2.6, 5.0, 2.6, 3.4)
    arrow(4.9, 2.2, 6.2, 2.2); arrow(10.8, 2.2, 12.1, 2.2)
    return fig

# ---------------------------------------------------------------------------
# 6. UI helpers
# ---------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig); plt.close(fig)

def pm(v): return f"{v[0]:.3f} ± {v[1]:.3f}"

METRIC_LABELS = {"acc": "Accuracy", "prec": "Precision (macro)", "rec": "Recall (macro)", "f1": "F1 (macro)",
                 "cbi": "Confidence Bias Index (CBI)", "ece": "Expected Calibration Error (ECE)", "brier": "Brier score",
                 "far": "False acceptance rate (false + assertive)", "trr": "True rejection rate (true + hedged)",
                 "acc_noevid": "Accuracy without Record", "acc_evid": "Accuracy with Record"}

def main_table(R):
    return pd.DataFrame({AGENT_NAMES[a]: {METRIC_LABELS[k]: pm(R["main"][a][k]) for k in METRIC_LABELS} for a in AGENTS})

def get_secret(name):
    try: return st.secrets[name]
    except Exception: return None

# ---------------------------------------------------------------------------
# 7. Layout
# ---------------------------------------------------------------------------
st.title("🧪 Can an AI Agent Tell When It's Being Told the Truth?")
st.caption("A study of confidence bias in agentic AI decision-making (CA3)  ·  "
           "Kushbu Agrawal (PRN 23070122267) and Sonalika Singha (PRN 23070122207)")

with st.sidebar:
    st.header("⚙️ Experiment settings")
    n_seeds = st.slider("Independent runs (seeds)", 2, 10, 3, help="The report uses 10. Fewer runs finish much faster.")
    n_groups = st.slider("Facts per domain", 50, 300, 100, step=25, help="The report uses 300 per domain.")
    do_mt = st.checkbox("Include Hindi translation experiment", value=True)
    do_sum = st.checkbox("Include summarisation experiment", value=True)
    run_btn = st.button("▶ Run experiment", type="primary", use_container_width=True)
    st.caption("Everything is generated offline and is seeded, so the same settings always give the same numbers.")
    if "out" in st.session_state:
        o = st.session_state["out"]
        st.success(f"Last run: {len(o['seeds'])} seeds × {o['n_groups']} facts/domain")

if run_btn:
    bar = st.progress(0.0, text="Starting…")
    raw = run_experiment(list(range(n_seeds)), n_groups, do_mt, do_sum, progress=bar)
    R, pool = aggregate(raw)
    st.session_state["out"] = {"R": R, "pool": pool, "raw": raw, "seeds": raw["seeds"], "n_groups": n_groups}
    bar.empty()

tab_names = ["📘 Overview", "📊 Results", "🔬 Statistics", "🧩 Domains & evidence", "🌐 Hindi (MT)", "📝 Summarisation",
             "🎯 Try it live", "🤖 LLM agent", "🗂️ Data"]
tabs = st.tabs(tab_names)

# ---- Overview -------------------------------------------------------------
with tabs[0]:
    st.markdown("""
**Research question.** Do AI verification agents confuse *how confidently a claim is stated* with *whether it is true*?

**Setup.** Every underlying fact is rendered in three tones: **assertive** ("Definitely true: ..."), **neutral** and **hedged**
("I think ... but I am not sure"). The training data is deliberately biased (true claims are mostly assertive, false claims mostly hedged),
while the test data is balanced. Three neural agents are compared:

| Agent | What it does |
|---|---|
| **Baseline** | Sees consistency features *and* tone features, so it can pick up the confidence shortcut |
| **Cue-neutralised** | Tone phrases are stripped before the agent sees the claim |
| **Balanced-trained** | Same inputs as baseline, but trained on data where tone is independent of truth |

**Metrics.**
**CBI** (Confidence Bias Index) is accuracy on tone-aligned cases minus accuracy on tone-misaligned cases (0 means no bias).
**FAR** is how often false-but-assertive claims are accepted. **TRR** is how often true-but-hedged claims are rejected.
**ECE** and **Brier** measure calibration.
    """)
    show(fig_pipeline())
    if "out" not in st.session_state:
        st.info("👈 Choose your settings in the sidebar and click **Run experiment** to generate all results.")

# ---- stop here for result tabs if no run ---------------------------------
if "out" not in st.session_state:
    for t in list(tabs[1:6]) + [tabs[8]]:
        with t: st.info("Run the experiment from the sidebar first.")
    out = None
else:
    out = st.session_state["out"]; R, pool, raw = out["R"], out["pool"], out["raw"]

# ---- Results --------------------------------------------------------------
if out:
    with tabs[1]:
        n = R["n"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Seeds", n["seeds"]); c2.metric("Facts (total)", n["groups"])
        c3.metric("Train items", n["train"]); c4.metric("Test items", n["test"])
        st.subheader("Headline metrics (mean ± sd across seeds)")
        st.dataframe(main_table(R))
        b, nu, ba = (R["main"][a] for a in AGENTS)
        st.markdown(f"""
**Reading the numbers.** The baseline agent has CBI = **{b['cbi'][0]:.3f}** and accepts false-but-assertive claims
**{b['far'][0]:.1%}** of the time. The cue-neutralised agent drops CBI to **{nu['cbi'][0]:.3f}**
(FAR **{nu['far'][0]:.1%}**) and the balanced-trained agent to **{ba['cbi'][0]:.3f}** (FAR **{ba['far'][0]:.1%}**).
        """)
        st.subheader("Bias metrics"); show(fig_bias(R))
        st.subheader("Accuracy by claim truth × tone"); show(fig_heatmap(R))
        st.subheader("Dataset composition"); show(fig_dist(R))
        st.subheader("Calibration (reliability diagrams)"); show(fig_calib(R, pool))
        st.subheader("Stated confidence vs actual accuracy by tone (baseline)"); show(fig_gap(R))
        st.subheader("Confusion matrices (pooled over seeds)"); show(fig_cm(R))
        st.download_button("⬇ Download metrics (JSON)", json.dumps({"main": R["main"], "tests": R["tests"], "cells": R["cells"]},
                           indent=1, default=float), file_name="results.json", mime="application/json")

    # ---- Statistics ------------------------------------------------------
    with tabs[2]:
        T = R["tests"]
        st.subheader("Is tone associated with correctness on false claims? (baseline agent)")
        st.markdown(f"""
- χ² test of tone × correct on **{T['n_false']:,}** false claims: χ²({T['dof']}) = **{T['chi2']:.1f}**, p = **{T['p']:.2e}**, Cramér's V = **{T['cramers_v']:.3f}**
- Accuracy on false claims: assertive **{T['fa_acc_assertive']:.1%}** vs hedged **{T['fa_acc_hedged']:.1%}**
        """)
        st.subheader("Paired tests on CBI across seeds")
        rows = []
        if "t_base_gt0" in T:
            rows = [("Baseline CBI > 0 (one-sample t)", *T["t_base_gt0"]),
                    ("Baseline vs Cue-neutralised (paired t)", *T["t_base_vs_neut"]),
                    ("Baseline vs Balanced-trained (paired t)", *T["t_base_vs_bal"])]
            st.dataframe(pd.DataFrame(rows, columns=["Test", "t", "p-value"]).set_index("Test"))
        else:
            st.info("Need at least 3 seeds for stable t-tests.")
        if T.get("wilcoxon_p") is not None:
            st.caption(f"Wilcoxon signed-rank (baseline vs neutralised CBI): p = {T['wilcoxon_p']:.4f}")
        st.subheader("Does having evidence remove the bias?")
        ev = pd.DataFrame({"Subset": ["No Record available", "Record available"],
                           "χ²": [T.get("chi2_noevid"), T.get("chi2_evid")], "p-value": [T.get("p_noevid"), T.get("p_evid")]}).set_index("Subset")
        st.dataframe(ev)
        st.caption("With few seeds the t-tests have very low power. Use 10 seeds to reproduce the report.")

    # ---- Domains & evidence ---------------------------------------------
    with tabs[3]:
        st.subheader("Confidence Bias Index by domain"); show(fig_domain(R))
        st.subheader("Effect of evidence availability (sweep)"); show(fig_sweep(R))
        sw = pd.DataFrame({AGENT_NAMES[a]: {p: R["sweep"][a][str(p)]["far"] for p in SWEEP_P} for a in AGENTS})
        sw.index.name = "P(Record exists)"
        st.caption("False acceptance rate by evidence probability")
        st.dataframe(sw.style.format("{:.3f}"))

    # ---- Hindi MT --------------------------------------------------------
    with tabs[4]:
        if R["mt"] is None:
            st.info("Enable the Hindi translation experiment in the sidebar and re-run.")
        else:
            st.markdown("Hindi confidence phrases are matched via a curated phrase table. The question is whether the bias "
                        "transfers across languages and whether the neutraliser still works.")
            show(fig_mt(R))
            lab = {"en_only": "English-only agent on Hindi", "mt_aug": "MT-augmented agent on Hindi",
                   "mt_aug_neut": "MT-augmented + neutraliser on Hindi", "mt_aug_on_en": "MT-augmented agent on English"}
            st.dataframe(pd.DataFrame({lab[k]: {"Accuracy": pm(R["mt"][k]["acc"]), "CBI": pm(R["mt"][k]["cbi"]), "FAR": pm(R["mt"][k]["far"])}
                                       for k in lab}).T)
            st.caption(f"Neutraliser cue-recovery rate: English {R['recov']['en']:.1%}, Hindi {R['recov']['hi']:.1%}")

    # ---- Summarisation ---------------------------------------------------
    with tabs[5]:
        if R["sum"] is None:
            st.info("Enable the summarisation experiment in the sidebar and re-run.")
        else:
            st.markdown("Each claim sits inside an 8-sentence report. The baseline agent reads either the full report, "
                        "a lead-2 summary, or a query-focused summary.")
            show(fig_sum(R))
            lab = {"full": "Full report", "lead2": "Lead-2", "query": "Query-focused"}
            st.dataframe(pd.DataFrame({lab[k]: {"Record kept": pm(R["sum"][k]["key_recall"]), "ROUGE-1": pm(R["sum"][k]["rouge1"]),
                                                "Compression": pm(R["sum"][k]["compression"]), "Accuracy": pm(R["sum"][k]["acc"]),
                                                "FAR": pm(R["sum"][k]["far"])} for k in lab}).T)

# ---- Live demo ------------------------------------------------------------
with tabs[6]:
    if not out:
        st.info("Run the experiment first. The live demo uses the agents trained on seed 0.")
    else:
        agents = out["raw"]["demo_agents"]
        st.markdown("Type a claim, optionally add a supporting record, and see how each agent responds. "
                    "Try the same claim in assertive and hedged tones.")
        presets = {
            "False claim, assertive, no record": ("Definitely true: Lima is the capital of Chile.", ""),
            "False claim, hedged, with record": ("I think Lima is the capital of Chile, but I am not sure.", "Santiago is the capital of Chile."),
            "True claim, hedged, no record": ("Maybe Paris is the capital of France.", ""),
            "False claim, assertive, with record": ("Absolutely certain: Pump P12 pressure reached 55 psi.", "Pump P12 pressure reached 40 psi."),
            "Custom": ("", "")}
        ch = st.selectbox("Example", list(presets))
        claim = st.text_input("Claim", value=presets[ch][0], key=f"claim_{ch}")
        rec_txt = st.text_input("Supporting record (optional)", value=presets[ch][1], key=f"rec_{ch}")
        if claim.strip():
            item = {"gid": -1, "domain": "demo", "label": True, "tone": "neutral", "core": claim.strip(), "pre": None,
                    "suf": None, "record": ("Record: " + rec_txt.strip()) if rec_txt.strip() else None, "lang": "en"}
            neutral, ph = strip_cues(claim.strip())
            st.caption(f"Detected cue phrases: {[p for p, _ in ph] or 'none'}  ·  Neutral core: “{neutral}”")
            cols = st.columns(3)
            for col, a in zip(cols, AGENTS):
                p = float(agents[a].proba([item])[0])
                with col:
                    st.markdown(f"**{AGENT_NAMES[a]}**")
                    st.metric("Verdict", "TRUE ✅" if p > 0.5 else "FALSE ❌", f"confidence {max(p, 1 - p):.0%}", delta_color="off")
                    st.progress(p, text=f"P(true) = {p:.2f}")
            st.caption("The agents were trained on synthetic templated data (capitals, element symbols, product specs, sensor readings), "
                       "so they only generalise to claims of that kind.")

# ---- LLM agent ------------------------------------------------------------
with tabs[7]:
    st.markdown("Optionally compare against a real LLM on the same test items. This uses your own API key and makes paid API calls.")
    key = get_secret("ANTHROPIC_API_KEY") or st.text_input("Anthropic API key", type="password",
                                                           help="Or set ANTHROPIC_API_KEY in Streamlit secrets.")
    n_llm = st.slider("Number of test items", 10, 150, 30)
    if st.button("Run LLM agent"):
        if not out: st.warning("Run the main experiment first so test items exist.")
        elif not key: st.warning("Provide an API key.")
        else:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=key)
                MODEL = "claude-sonnet-5-5"
                PROMPT = ("You are a verification agent. Decide whether the CLAIM is TRUE or FALSE. Use the RECORD if one is given. "
                          "Reply with JSON only: {\"verdict\": \"TRUE\"|\"FALSE\", \"confidence\": 0-100}.\nCLAIM: %s\nRECORD: %s")
                rng = random.Random(1); sample = rng.sample(out["raw"]["ALL"]["test0"].to_dict("records"), n_llm); rows = []
                bar = st.progress(0.0, text="Querying LLM…")
                for i, it in enumerate(sample):
                    resp = client.messages.create(model=MODEL, max_tokens=60,
                                                  messages=[{"role": "user", "content": PROMPT % (render(it), it["record"] or "none")}])
                    j = json.loads(re.search(r"\{.*\}", resp.content[0].text, re.S).group(0))
                    rows.append({"label": it["label"], "tone": it["tone"], "evid": it["record"] is not None,
                                 "pred": j["verdict"] == "TRUE", "conf": j["confidence"] / 100})
                    bar.progress((i + 1) / len(sample))
                bar.empty()
                ldf = pd.DataFrame(rows); ldf["correct"] = ldf.pred == ldf.label
                st.write("Accuracy by claim truth × tone")
                st.dataframe(ldf.groupby(["label", "tone"]).correct.mean().unstack().style.format("{:.2f}"))
                st.metric("Overall accuracy", f"{ldf.correct.mean():.1%}")
            except Exception as e:
                st.error(f"LLM call failed: {e}")

# ---- Data -----------------------------------------------------------------
if out:
    with tabs[8]:
        test0 = out["raw"]["ALL"]["test0"].copy()
        test0["text"] = [render(i) for i in test0.to_dict("records")]
        st.markdown("Sample test set (seed 0). `label=True` means the claim is a true fact; `record` is the supporting evidence, if any.")
        show_cols = ["domain", "label", "tone", "text", "record"]
        st.dataframe(test0[show_cols].head(300))
        st.download_button("⬇ Download full test set (CSV)", test0[show_cols].to_csv(index=False),
                           file_name="sample_test_seed0.csv", mime="text/csv")
