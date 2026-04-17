"""Graph construction + path enumeration (Phase 3).

No networkx dependency on hot path -- we store edges as a dict-of-dicts and
enumerate 2/3-hop paths directly over the adjacency list.
"""
from __future__ import annotations
import math, random
from collections import defaultdict
from itertools import combinations
from typing import Iterable


def build_inverted_index(docs: list[dict]) -> dict[str, set[str]]:
    """entity -> set(doc_id)."""
    inv: dict[str, set[str]] = defaultdict(set)
    for d in docs:
        did = d["doc_id"]
        for l in d.get("links", []):
            t = l.get("href")
            if t:
                inv[t].add(did)
    return inv


def doc_title_index(docs: list[dict]) -> dict[str, str]:
    """lowercased title -> doc_id."""
    out = {}
    for d in docs:
        t = (d.get("title") or "").lower().strip()
        if t:
            out[t] = d["doc_id"]
    return out


def idf_weights(inv: dict[str, set[str]], n_docs: int,
                min_df: int = 2, generic_percentile: float = 0.98) -> dict[str, float]:
    """Return IDF per entity; drop entities above the `generic_percentile` doc-freq cutoff."""
    dfs = sorted((len(v) for v in inv.values()), reverse=False)
    if not dfs:
        return {}
    # percentile cutoff -> generic if df > cutoff_val
    idx = max(0, min(len(dfs) - 1, int(len(dfs) * generic_percentile)))
    cutoff = dfs[idx]
    out = {}
    for e, ds in inv.items():
        df = len(ds)
        if df < min_df:
            continue
        if df > cutoff and df >= 100:
            continue
        out[e] = math.log(n_docs / df)
    return out


def build_edges(docs: list[dict], inv: dict[str, set[str]],
                idf: dict[str, float], title_idx: dict[str, str]
                ) -> dict[tuple[str, str], dict]:
    """Edge dict keyed by sorted doc-pair tuple.

    edge fields: weight (sum of idf), shared_entities (list[str]), direct (bool).
    """
    edges: dict[tuple[str, str], dict] = {}
    # shared-entity edges
    for e, ds in inv.items():
        if e not in idf:
            continue
        w_e = idf[e]
        ds_list = list(ds)
        if len(ds_list) < 2 or len(ds_list) > 100:
            continue
        for a, b in combinations(ds_list, 2):
            k = (a, b) if a < b else (b, a)
            ed = edges.get(k)
            if ed is None:
                edges[k] = {"weight": w_e, "shared_entities": [e], "direct": False}
            else:
                ed["weight"] += w_e
                ed["shared_entities"].append(e)
    # direct edges (A links to B's title)
    did_by_doc: dict[str, set[str]] = {d["doc_id"]: set() for d in docs}
    for d in docs:
        for l in d.get("links", []):
            t = l.get("href")
            if t and t in title_idx:
                tgt = title_idx[t]
                if tgt != d["doc_id"]:
                    did_by_doc[d["doc_id"]].add(tgt)
    for a, out_set in did_by_doc.items():
        for b in out_set:
            k = (a, b) if a < b else (b, a)
            ed = edges.get(k)
            # Title idf: use max of both directions' title idf if known
            t_a_idf = 1.0
            ta = next((x for x in idf if x == a), None)  # unlikely
            tb = next((x for x in idf if x == b), None)
            add = 2.0 * (idf.get(ta, 1.0) if ta else 1.0) if False else 2.0  # simple boost
            if ed is None:
                edges[k] = {"weight": add, "shared_entities": [], "direct": True}
            else:
                ed["direct"] = True
                ed["weight"] += add
    return edges


def adjacency(edges: dict[tuple[str, str], dict]) -> dict[str, dict[str, dict]]:
    """adj[a][b] = edge dict."""
    adj: dict[str, dict[str, dict]] = defaultdict(dict)
    for (a, b), ed in edges.items():
        adj[a][b] = ed
        adj[b][a] = ed
    return adj


def infer_reasoning_type(a_doc: dict, b_doc: dict, shared: list[str],
                         idf: dict[str, float]) -> str:
    """Cheap heuristic on title + shared entities to guess reasoning type.
    See CLAUDE.md 6.3 'Reasoning-type inference heuristics'."""
    a_title = (a_doc.get("title") or "").lower()
    b_title = (b_doc.get("title") or "").lower()
    a_wc = len((a_doc.get("clean_text") or "").split())
    b_wc = len((b_doc.get("clean_text") or "").split())

    # definition_application: one doc is very short (definitional), the other long
    if (a_wc < 400 and b_wc >= 600) or (b_wc < 400 and a_wc >= 600):
        return "definition_application"

    # comparison: endpoint titles share strong category token AND share specific attribute
    PERSON_HINTS = {"scientist", "physicist", "chemist", "author", "actor", "philosopher",
                    "writer", "director", "artist", "composer", "musician", "poet"}
    ORG_HINTS = {"company", "corporation", "ltd", "inc", "group", "holdings"}
    COUNTRY_HINTS = {"republic", "kingdom", "state", "federation"}

    # We can't easily tell person-vs-person from title; fall back to shared-entity cues
    sentities_low = [s for s in shared if s in idf and idf[s] > 3.0]  # moderately specific
    if len(sentities_low) >= 2:
        # multiple specific shared entities -> comparison/bridge
        ent_tokens = {tok for s in shared for tok in s.split()}
        if ent_tokens & {"prize", "award", "medal", "ceo", "president", "founder",
                         "winner", "recipient"}:
            return "comparison"

    # temporal_chain: shared entity with year words in both
    YEAR = __import__("re").compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
    if shared:
        a_years = set(YEAR.findall((a_doc.get("clean_text") or "")[:4000]))
        b_years = set(YEAR.findall((b_doc.get("clean_text") or "")[:4000]))
        if a_years and b_years and a_years != b_years:
            return "temporal_chain"

    # cause_effect: verb polarity heuristic on first 500 chars of each
    CAUSE = {"caused", "triggered", "resulted", "led", "sparked", "provoked",
             "after", "because", "due"}
    at = (a_doc.get("clean_text") or "")[:500].lower()
    bt = (b_doc.get("clean_text") or "")[:500].lower()
    if any(w in at for w in CAUSE) and any(w in bt for w in CAUSE):
        return "cause_effect"

    return "bridge_entity"


def score_path(path_docs: list[dict], shared: list[str], bridge: str,
               idf: dict[str, float], min_spec: float, max_spec: float,
               prefer_indirect: bool) -> float:
    """Return a 0..1 path-quality score."""
    if not shared:
        return 0.0
    # specificity of bridge
    b_idf = idf.get(bridge, 1.0)
    max_idf = max(idf.values()) if idf else 10.0
    spec = b_idf / max_idf if max_idf else 0.0
    if not (min_spec <= spec <= max_spec):
        spec_factor = 0.5
    else:
        spec_factor = 1.0
    # doc complementarity via word-count diversity
    wcs = [len((d.get("clean_text") or "").split()) for d in path_docs]
    if len(wcs) >= 2:
        div = min(wcs) / max(wcs) if max(wcs) > 0 else 0.0
    else:
        div = 0.5
    # avoid bridge being title of either endpoint
    indirect = 1.0
    if prefer_indirect:
        titles = {(d.get("title") or "").lower() for d in path_docs}
        if bridge in titles:
            indirect = 0.4
    s = 0.45 * spec_factor + 0.25 * indirect + 0.15 * div + 0.15 * min(1.0, len(shared) / 4.0)
    return max(0.0, min(1.0, s))


def enumerate_2hop(adj: dict[str, dict[str, dict]], max_per_node: int = 100,
                   seed: int = 17) -> list[tuple[str, str, dict]]:
    """Return (a, b, edge) direct pairs."""
    rng = random.Random(seed)
    out = []
    seen = set()
    nodes = list(adj.keys())
    rng.shuffle(nodes)
    for a in nodes:
        nbrs = list(adj[a].items())
        rng.shuffle(nbrs)
        cnt = 0
        for b, ed in nbrs:
            k = (a, b) if a < b else (b, a)
            if k in seen:
                continue
            seen.add(k)
            out.append((k[0], k[1], ed))
            cnt += 1
            if cnt >= max_per_node:
                break
    return out


def enumerate_3hop(adj: dict[str, dict[str, dict]], max_paths: int = 5000,
                   seed: int = 19) -> list[tuple[str, str, str, dict, dict]]:
    """Return random a -> m -> b triples (a != b). Sample up to max_paths."""
    rng = random.Random(seed)
    out = []
    nodes = list(adj.keys())
    rng.shuffle(nodes)
    for a in nodes:
        mids = list(adj[a].items())
        if not mids:
            continue
        rng.shuffle(mids)
        for m, ed_am in mids[:4]:
            for b, ed_mb in list(adj[m].items())[:4]:
                if b == a or b == m:
                    continue
                if b in adj[a]:
                    # skip trivially 2-hop
                    continue
                out.append((a, m, b, ed_am, ed_mb))
                if len(out) >= max_paths:
                    return out
    return out
