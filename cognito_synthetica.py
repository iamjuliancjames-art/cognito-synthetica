# =============================================================================
#  Cognito Synthetica v0.9 — Martian + Seeker + Dreamer (Somnambulist) unified
#  Pure stdlib Python 3.11+ (NO dependencies)
#
#  Concept:
#    - Martian: persistent cognitive manifold memory (rooms, anchors, attractors,
#               stability/importance/novelty, graph, reflect, prune)
#    - Seeker:  IR engine for "rooms = documents/page results" with fielded BM25,
#               phrase search, bigram index, geodesic graph expansion, MMR diversity
#    - Dreamer: background sleepwalker that rehearses + triggers consolidation
#
#  Unification:
#    - One shared RoomStore (rooms + graph + meta + provenance)
#    - Dual indexing:
#        * Martian retrieval: similarity + meta + graph proximity
#        * Seeker retrieval: BM25 fielded + phrase/bigram + graph geodesic + MMR
#    - Dreamer drives periodic reflect cycles:
#        * consolidates "pudding" (low-stability) into semantic hubs (non-destructive)
#
#  Notes:
#    - "Pseudo similarity" used for graph edges and MMR (stdlib stand-in for embeddings)
#    - Reflect archives sources instead of deleting; provenance links preserved
# =============================================================================

import math
import time
import re
import random
import heapq
from collections import defaultdict, Counter, deque
from typing import Dict, List, Optional, Set, Tuple


# =============================================================================
#  Shared Utilities
# =============================================================================

_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "it","this","that","as","at","by","from","be","been","not","no","but","so","if","then",
    "than","into","about"
}

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# =============================================================================
#  RoomStore — shared state: rooms + graph + provenance + pruning
# =============================================================================

class RoomStore:
    """
    Canonical storage for rooms and the Lotus-weighted graph.
    Engines (Martian/Seeker) share this store.
    """

    def __init__(self, max_rooms: int = 800, sim_threshold: float = 0.25, graph_neighbors: int = 8):
        self.rooms: List[Dict] = []
        self.room_id_counter = 0

        self.max_rooms = max_rooms
        self.sim_threshold = sim_threshold
        self.graph_neighbors = graph_neighbors

        # Graph: rid -> {neighbor_rid: lotus_cost}
        self.graph: Dict[int, Dict[int, float]] = defaultdict(dict)

        # Access order (LRU-ish signal)
        self.access_order = deque(maxlen=max_rooms * 2)

        # Continuity mechanisms
        self.anchor_ids: Set[int] = set()     # identity anchors
        self.attractors: List[str] = []       # goal gravity / "stable vibes"
        self.recent_texts = deque(maxlen=80)  # for novelty and drift heuristics

        # Lotus knobs
        self.EPS = 1e-10
        self.LAMBDA_PI = 0.30
        self.MU_RISK = 0.60
        self.SINGULARITY_GATE = 0.80

    # ──────────────────────────────────────────────────────────────────────────
    # Text processing + pseudo similarity
    # ──────────────────────────────────────────────────────────────────────────
    def tokens(self, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[a-z0-9']+", text.lower())
        return [t for t in toks if t not in _STOP and len(t) >= 2]

    def pseudo_sim(self, a: str, b: str) -> float:
        """
        Stdlib stand-in for embedding cosine similarity:
        n-gram Jaccard + length ratio shaping.
        """
        if not a or not b:
            return 0.0
        a, b = a.lower(), b.lower()

        def ngrams(s: str, n: int) -> Set[str]:
            if len(s) < n:
                return set()
            return {s[i:i+n] for i in range(len(s)-n+1)}

        def jacc(x: Set[str], y: Set[str]) -> float:
            if not x and not y:
                return 0.0
            return len(x & y) / max(1, len(x | y))

        a3, b3 = ngrams(a, 3), ngrams(b, 3)
        a4, b4 = ngrams(a, 4), ngrams(b, 4)
        ov = max(jacc(a3, b3), jacc(a4, b4), 0.0)

        len_r = min(len(a), len(b)) / max(1, max(len(a), len(b)))
        return ov * (0.35 + 0.65 * (len_r ** 1.25))

    def nuance(self, text: str) -> float:
        toks = self.tokens(text)
        return (len(set(toks)) / len(toks)) if toks else 0.0

    def novelty(self, text: str, lookback: int = 80) -> float:
        if not self.rooms:
            return 1.0
        recent = list(self.recent_texts)[-min(len(self.recent_texts), lookback):]
        max_sim = 0.0
        for t in recent:
            max_sim = max(max_sim, self.pseudo_sim(text, t))
        return _clamp(1.0 - max_sim, 0.0, 1.0)

    def lotus_cost(self, dist: float, pi_a: float, pi_b: float, risk_a: float, risk_b: float) -> float:
        pi = 0.5 * (pi_a + pi_b)
        risk = max(risk_a, risk_b)
        pi_term = self.LAMBDA_PI * pi
        risk_term = self.MU_RISK * risk
        sing = (1.0 / max(self.EPS, (1.0 - risk))) if risk > self.SINGULARITY_GATE else 0.0
        return dist + pi_term + risk_term + sing

    # ──────────────────────────────────────────────────────────────────────────
    # Room lifecycle
    # ──────────────────────────────────────────────────────────────────────────
    def room_by_id(self, rid: int) -> Optional[Dict]:
        for r in self.rooms:
            if r["id"] == rid:
                return r
        return None

    def add_room(
        self,
        canonical: str,
        kind: str,
        fields: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict] = None,
        is_anchor: bool = False,
        attractor: bool = False,
    ) -> int:
        canonical = (canonical or "").strip()
        if not canonical:
            return -1

        # near-duplicate skip among recent actives
        max_sim = 0.0
        for r in self.rooms[-min(len(self.rooms), 140):]:
            if r["meta"].get("archived"):
                continue
            max_sim = max(max_sim, self.pseudo_sim(canonical, r.get("canonical", "")))
        if max_sim > 0.97:
            return -1

        rid = self.room_id_counter
        self.room_id_counter += 1
        ts = time.time()

        novelty = self.novelty(canonical)
        nuance = self.nuance(canonical)

        kind_bias = {"semantic": 0.45, "commitment": 0.35, "state": 0.25, "doc": 0.20, "page": 0.15, "snippet": 0.05}.get(kind, 0.0)
        stability = _clamp(_sigmoid(-0.55 + 1.10 * novelty + 1.70 * nuance + kind_bias), 0.05, 1.0)

        age_h = 0.0
        recency = 1.0  # just added
        length_term = min(1.0, len(canonical.split()) / 160.0)
        novelty_term = min(1.0, novelty / 0.8)
        importance = _clamp(0.45 * recency + 0.30 * length_term + 0.25 * novelty_term, 0.02, 1.0)

        pi = round(random.random(), 4)
        risk = round(random.random() * 0.6, 4)

        meta = {
            "kind": kind,
            "ts": ts,
            "novelty": round(novelty, 4),
            "nuance": round(nuance, 4),
            "stability": round(stability, 4),
            "importance": round(importance, 4),
            "pi": pi,
            "risk": risk,
            "archived": False,
        }
        if metadata:
            meta.update(metadata)

        room = {
            "id": rid,
            "canonical": canonical,
            "fields": fields or {},
            "meta": meta,
            "links": {"sources": [], "hubs": []},
        }

        self.rooms.append(room)
        self.access_order.append(rid)
        self.recent_texts.append(canonical)

        if is_anchor:
            self.anchor_ids.add(rid)
        if attractor:
            self.attractors.append(canonical)

        self._connect_room(rid)

        # capacity enforcement is delegated to Cognito (so index removal can happen there)
        return rid

    def _connect_room(self, rid: int):
        r = self.room_by_id(rid)
        if not r or r["meta"].get("archived"):
            return

        sims: List[Tuple[float, int]] = []
        for other in self.rooms[-min(len(self.rooms), 300):]:
            oid = other["id"]
            if oid == rid or other["meta"].get("archived"):
                continue
            s = self.pseudo_sim(r["canonical"], other["canonical"])
            sims.append((s, oid))
        sims.sort(reverse=True)

        for sim_val, oid in sims[: self.graph_neighbors]:
            if sim_val < self.sim_threshold:
                continue
            o = self.room_by_id(oid)
            if not o:
                continue
            dist = 1.0 - sim_val
            cost = self.lotus_cost(dist, r["meta"]["pi"], o["meta"]["pi"], r["meta"]["risk"], o["meta"]["risk"])
            cost = round(cost, 6)
            self.graph[rid][oid] = cost
            self.graph[oid][rid] = cost

    def remove_room(self, rid: int):
        # remove from room list
        self.rooms = [r for r in self.rooms if r["id"] != rid]
        self.anchor_ids.discard(rid)

        # remove from graph
        self.graph.pop(rid, None)
        for neigh in self.graph.values():
            neigh.pop(rid, None)

    def status(self) -> str:
        edges = sum(len(v) for v in self.graph.values()) // 2
        archived = sum(1 for r in self.rooms if r["meta"].get("archived"))
        kinds = Counter(r["meta"]["kind"] for r in self.rooms)
        return (
            f"RoomStore: rooms={len(self.rooms)}/{self.max_rooms} archived={archived} "
            f"anchors={len(self.anchor_ids)} attractors={len(self.attractors)} edges={edges} kinds={dict(kinds)}"
        )


# =============================================================================
#  Seeker Index — fielded inverted index + BM25 + bigram + phrase matching
# =============================================================================

class SeekerIndex:
    """
    Fielded IR index over RoomStore rooms where kind in {"doc","page","snippet","semantic"}.
    Supports:
      - fielded BM25
      - bigram BM25
      - quoted phrase exact bonus
      - candidate gen via inverted + bigram inverted
    """

    def __init__(self, store: RoomStore):
        self.store = store

        # rid -> field -> Counter(term)
        self.tf: Dict[int, Dict[str, Counter]] = {}
        self.dl: Dict[int, Dict[str, int]] = {}
        self.avgdl: Dict[str, float] = defaultdict(float)

        self.df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.inverted: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

        self.bigram_tf: Dict[int, Dict[str, Counter]] = {}
        self.bigram_df: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.bigram_inverted: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))

        # BM25 params
        self.k1 = 1.2
        self.b = 0.75

        # scoring knobs
        self.field_weights = {"title": 2.2, "snippet": 1.4, "body": 1.0, "tags": 1.8}
        self.bigram_boost = 0.20
        self.phrase_boost = 0.55

        self.EPS = 1e-10

    def _extract_quoted_phrases(self, query: str) -> List[str]:
        return [p.strip() for p in re.findall(r'"([^"]+)"', query) if p.strip()]

    def _bigrams(self, toks: List[str]) -> List[str]:
        if len(toks) < 2:
            return []
        return [toks[i] + "_" + toks[i+1] for i in range(len(toks)-1)]

    def _recompute_avgdl(self):
        counts = defaultdict(int)
        totals = defaultdict(int)
        for rid, fields in self.dl.items():
            for f, L in fields.items():
                if L > 0:
                    counts[f] += 1
                    totals[f] += L
        for f in set(list(self.field_weights.keys()) + list(totals.keys())):
            self.avgdl[f] = (totals[f] / counts[f]) if counts[f] else 0.0

    def index_room(self, rid: int):
        r = self.store.room_by_id(rid)
        if not r:
            return
        kind = r["meta"]["kind"]
        if kind not in ("doc", "page", "snippet", "semantic"):
            return

        fields = r.get("fields", {})
        for f in ("title", "snippet", "body", "tags"):
            txt = fields.get(f, "") or ""
            toks = self.store.tokens(txt)
            if not toks:
                continue

            tf = Counter(toks)
            self.tf.setdefault(rid, {})[f] = tf
            self.dl.setdefault(rid, {})[f] = sum(tf.values())

            for term in tf.keys():
                if rid not in self.inverted[f][term]:
                    self.inverted[f][term].add(rid)
                    self.df[f][term] += 1

            bigs = self._bigrams(toks)
            btf = Counter(bigs)
            self.bigram_tf.setdefault(rid, {})[f] = btf
            for bg in btf.keys():
                if rid not in self.bigram_inverted[f][bg]:
                    self.bigram_inverted[f][bg].add(rid)
                    self.bigram_df[f][bg] += 1

        self._recompute_avgdl()

    def remove_room(self, rid: int):
        tf_fields = self.tf.pop(rid, None)
        btf_fields = self.bigram_tf.pop(rid, None)
        self.dl.pop(rid, None)

        if tf_fields:
            for field, tf in tf_fields.items():
                for term in tf.keys():
                    postings = self.inverted[field].get(term)
                    if postings and rid in postings:
                        postings.remove(rid)
                        self.df[field][term] = max(0, self.df[field][term] - 1)
                        if not postings:
                            self.inverted[field].pop(term, None)
                            self.df[field].pop(term, None)

        if btf_fields:
            for field, btf in btf_fields.items():
                for bg in btf.keys():
                    postings = self.bigram_inverted[field].get(bg)
                    if postings and rid in postings:
                        postings.remove(rid)
                        self.bigram_df[field][bg] = max(0, self.bigram_df[field][bg] - 1)
                        if not postings:
                            self.bigram_inverted[field].pop(bg, None)
                            self.bigram_df[field].pop(bg, None)

        self._recompute_avgdl()

    def _idf(self, field: str, term: str) -> float:
        N = max(1, len(self.store.rooms))
        df = self.df[field].get(term, 0)
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _bigram_idf(self, field: str, bg: str) -> float:
        N = max(1, len(self.store.rooms))
        df = self.bigram_df[field].get(bg, 0)
        return math.log(1.0 + (N - df + 0.5) / (df + 0.5))

    def _bm25_field_score(self, rid: int, field: str, q_terms: List[str]) -> float:
        tf = self.tf.get(rid, {}).get(field)
        if not tf:
            return 0.0
        dl = self.dl.get(rid, {}).get(field, 0)
        avgdl = self.avgdl.get(field, 0.0)
        if dl <= 0 or avgdl <= 0:
            return 0.0

        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f <= 0:
                continue
            idf = self._idf(field, term)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score += idf * (f * (self.k1 + 1)) / (denom + self.EPS)
        return score

    def _bigram_field_score(self, rid: int, field: str, q_bigrams: List[str]) -> float:
        btf = self.bigram_tf.get(rid, {}).get(field)
        if not btf:
            return 0.0
        dl = self.dl.get(rid, {}).get(field, 0)
        avgdl = self.avgdl.get(field, 0.0)
        if dl <= 0 or avgdl <= 0:
            return 0.0

        score = 0.0
        for bg in q_bigrams:
            f = btf.get(bg, 0)
            if f <= 0:
                continue
            idf = self._bigram_idf(field, bg)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / avgdl))
            score += idf * (f * (self.k1 + 1)) / (denom + self.EPS)
        return score

    def _phrase_bonus(self, canonical: str, phrases: List[str]) -> float:
        if not phrases:
            return 0.0
        hay = canonical.lower()
        hits = 0
        for p in phrases:
            if p.lower() in hay:
                hits += 1
        return hits * self.phrase_boost

    def candidate_set(self, q_terms: List[str], q_bigrams: List[str]) -> Set[int]:
        cands: Set[int] = set()
        for term in set(q_terms):
            for field in self.field_weights.keys():
                cands |= self.inverted[field].get(term, set())
        for bg in set(q_bigrams):
            for field in self.field_weights.keys():
                cands |= self.bigram_inverted[field].get(bg, set())
        return cands

    def score_candidates(self, query: str, add_sim_rerank: bool = True) -> Dict[int, float]:
        phrases = self._extract_quoted_phrases(query)
        q_terms = self.store.tokens(query)

        q_bigrams = []
        q_bigrams.extend(self._bigrams(q_terms))
        for p in phrases:
            pt = self.store.tokens(p)
            q_bigrams.extend(self._bigrams(pt))

        cands = self.candidate_set(q_terms, q_bigrams)
        if not cands:
            return {}

        now = time.time()
        base_scores: Dict[int, float] = {}

        for rid in cands:
            r = self.store.room_by_id(rid)
            if not r or r["meta"].get("archived"):
                continue

            bm = 0.0
            bg = 0.0
            for field, w in self.field_weights.items():
                bm += w * self._bm25_field_score(rid, field, q_terms)
                if q_bigrams:
                    bg += w * self._bigram_field_score(rid, field, q_bigrams)

            phr = self._phrase_bonus(r["canonical"], phrases)

            age_days = (now - r["meta"]["ts"]) / 86400.0
            recency = 1.0 / (1.0 + age_days)

            score = (
                1.00 * bm +
                self.bigram_boost * bg +
                phr +
                0.10 * self.store_kind_priority(r["meta"]["kind"]) +
                0.08 * r["meta"]["importance"] +
                0.06 * r["meta"]["stability"] +
                0.05 * recency
            )

            if add_sim_rerank:
                score += 0.18 * self.store.pseudo_sim(query, r["canonical"])

            base_scores[rid] = score

        return base_scores

    def store_kind_priority(self, kind: str) -> float:
        return {
            "semantic": 1.00, "doc": 0.85, "page": 0.82, "snippet": 0.70, "unknown": 0.40,
            "commitment": 0.90, "state": 0.75, "episodic": 0.55
        }.get(kind, 0.40)


# =============================================================================
#  Martian Engine — continuity memory retrieval + drift checks + reflect
# =============================================================================

class MartianEngine:
    """
    Uses RoomStore only (no inverted index).
    Retrieval is similarity + meta + graph proximity.
    Reflect consolidates low-stability recent rooms into semantic hub (archives sources).
    """

    def __init__(self, store: RoomStore):
        self.store = store
        self.history_window = 40
        self.recent_texts = deque(maxlen=self.history_window)

        # Retrieval weights
        self.W_SIM = 0.45
        self.W_KIND = 0.18
        self.W_IMP = 0.12
        self.W_STAB = 0.10
        self.W_REC = 0.10
        self.W_GRAPH = 0.05

    def kind_priority(self, kind: str) -> float:
        return {
            "semantic": 1.00,
            "commitment": 0.85,
            "state": 0.75,
            "episodic": 0.55,
            "doc": 0.70,
            "page": 0.70,
            "snippet": 0.65,
            "unknown": 0.40,
        }.get(kind, 0.40)

    def talos_check(self, new_text: str) -> Dict:
        self.recent_texts.append((new_text or "").lower())
        if len(self.recent_texts) < 5:
            return {"stable": True, "nudge_suggestion": None}

        words = []
        for t in self.recent_texts:
            words.extend(re.findall(r"[a-z0-9']+", t))

        if not words:
            return {"stable": True, "nudge_suggestion": None}

        cnt = Counter(words)
        total = len(words)
        ent = -sum((c/total) * math.log2((c/total) + self.store.EPS) for c in cnt.values())

        repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        coherence = 1.0 - min(0.95, repeats / max(1, (total - 1)))

        drift = (ent < 2.6) or (coherence < 0.50)

        nudge = None
        if drift and self.store.attractors:
            nudge = f"Pull toward attractor: {self.store.attractors[-1][:120]}…"

        return {"stable": not drift, "entropy": round(ent, 3), "coherence_proxy": round(coherence, 3), "nudge_suggestion": nudge}

    def retrieve(self, query: str, top_k: int = 6, min_sim: float = 0.20, expand_hops: int = 1) -> List[Dict]:
        if not query or not self.store.rooms:
            return []

        now = time.time()
        base_scores: Dict[int, float] = {}

        for r in self.store.rooms:
            if r["meta"].get("archived"):
                continue
            sim = self.store.pseudo_sim(query, r["canonical"])
            if sim < min_sim:
                continue

            age_days = (now - r["meta"]["ts"]) / 86400.0
            recency = 1.0 / (1.0 + age_days)

            score = (
                self.W_SIM * sim +
                self.W_KIND * self.kind_priority(r["meta"]["kind"]) +
                self.W_IMP * r["meta"]["importance"] +
                self.W_STAB * r["meta"]["stability"] +
                self.W_REC * recency
            )
            if r["id"] in self.store.anchor_ids:
                score += 0.05

            base_scores[r["id"]] = score

        if not base_scores:
            return []

        expanded = dict(base_scores)
        if expand_hops >= 1:
            seeds = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]
            for seed_id, seed_score in seeds:
                for nb, cost in self.store.graph.get(seed_id, {}).items():
                    nb_room = self.store.room_by_id(nb)
                    if not nb_room or nb_room["meta"].get("archived"):
                        continue
                    proximity = 1.0 / (1.0 + cost)
                    bonus = self.W_GRAPH * proximity * (0.6 + 0.4 * seed_score)
                    expanded[nb] = max(expanded.get(nb, 0.0), seed_score * 0.35 + bonus)

        ranked = sorted(expanded.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for rid, _ in ranked:
            rr = self.store.room_by_id(rid)
            if rr:
                out.append(rr)
                self.store.access_order.append(rid)
        return out

    def reflect(self, recent_hours: float = 72.0, min_cluster: int = 6, max_sources: int = 24) -> Optional[int]:
        """
        Consolidate low-stability recent rooms (episodic/state/unknown/page/snippet)
        into a semantic hub and archive sources (non-destructive).
        """
        now = time.time()
        horizon = recent_hours * 3600.0

        candidates = []
        for r in self.store.rooms:
            m = r["meta"]
            if m.get("archived"):
                continue
            if (now - m["ts"]) > horizon:
                continue
            if m["kind"] not in ("episodic", "state", "unknown", "page", "snippet", "doc"):
                continue
            if m["stability"] > 0.70:
                continue
            candidates.append(r)

        if len(candidates) < min_cluster:
            return None

        # center selection: most similar to others
        best_center = None
        best_score = -1.0
        for r in candidates:
            sims = []
            for o in candidates:
                if o["id"] == r["id"]:
                    continue
                sims.append(self.store.pseudo_sim(r["canonical"], o["canonical"]))
            if not sims:
                continue
            score = sum(sorted(sims, reverse=True)[:min(10, len(sims))]) / max(1, min(10, len(sims)))
            if score > best_score:
                best_score = score
                best_center = r

        if not best_center:
            return None

        center = best_center["canonical"]
        members = []
        for r in candidates:
            s = self.store.pseudo_sim(center, r["canonical"])
            if s >= max(self.store.sim_threshold, 0.28):
                members.append((s, r))
        members.sort(reverse=True, key=lambda x: x[0])
        members = [r for _, r in members[:max_sources]]

        if len(members) < min_cluster:
            return None

        hub_title, hub_body, hub_tags = self._summarize_cluster(members)
        hub_fields = {"title": hub_title, "body": hub_body, "snippet": "", "tags": " ".join(hub_tags)}
        hub_canon = "\n".join([hub_title, hub_body, " ".join(hub_tags)]).strip()

        hub_id = self.store.add_room(hub_canon, kind="semantic", fields=hub_fields, metadata={"source": "reflect"}, is_anchor=False, attractor=False)
        hub = self.store.room_by_id(hub_id)
        if not hub:
            return None

        hub["links"]["sources"] = [m["id"] for m in members]

        for m in members:
            m["links"]["hubs"].append(hub_id)
            m["meta"]["archived"] = True

            # strong hub-member edge
            dist = 0.20
            cost = self.store.lotus_cost(dist, hub["meta"]["pi"], m["meta"]["pi"], hub["meta"]["risk"], m["meta"]["risk"])
            self.store.graph[hub_id][m["id"]] = round(cost, 6)
            self.store.graph[m["id"]][hub_id] = round(cost, 6)

        return hub_id

    def _summarize_cluster(self, members: List[Dict]) -> Tuple[str, str, List[str]]:
        words = []
        for m in members:
            words += self.store.tokens(m["canonical"])
        cnt = Counter(words)
        tags = [w for w, _ in cnt.most_common(10)] or ["hub"]

        exemplars = []
        for m in members[:6]:
            t = (m.get("fields", {}).get("title") or "").strip()
            if not t:
                t = m["canonical"].replace("\n", " ")[:80] + ("…" if len(m["canonical"]) > 80 else "")
            exemplars.append(t)

        title = f"Cognito hub ({len(members)} sources): " + ", ".join(tags[:4])
        body = "Consolidated semantic hub.\n" + f"Keywords: {', '.join(tags[:8])}\n" + "Exemplars:\n- " + "\n- ".join(exemplars)
        return title, body, tags[:8]


# =============================================================================
#  Dreamer — Somnambulist / Sleepwalker
# =============================================================================

class Dreamer:
    """
    Periodic rehearsal + consolidation driver.
    Call .tick() per loop (or per user message).
    """

    def __init__(self, store: RoomStore, martian: MartianEngine, reflect_every: int = 8):
        self.store = store
        self.martian = martian
        self.dream_level = 0
        self.reflect_every = max(1, reflect_every)
        self._ticks = 0

    def tick(self) -> Optional[int]:
        self.dream_level += 1
        self._ticks += 1

        # rehearsal: touch a couple random active rooms to bias access order
        active = [r for r in self.store.rooms if not r["meta"].get("archived")]
        if active:
            for _ in range(min(2, len(active))):
                r = random.choice(active)
                self.store.access_order.append(r["id"])

        # scheduled consolidation
        if (self._ticks % self.reflect_every) == 0:
            return self.martian.reflect()
        return None


# =============================================================================
#  Cognito Synthetica — orchestrator (one code base)
# =============================================================================

class CognitoSynthetica:
    """
    Unified cognitive system:
      - ingest memory rooms (Martian style)
      - ingest documents/page results (Seeker style)
      - retrieve via either mode
      - dream ticks consolidate periodically
      - enforce capacity with index-safe pruning
    """

    def __init__(self, max_rooms: int = 800, sim_threshold: float = 0.25):
        self.store = RoomStore(max_rooms=max_rooms, sim_threshold=sim_threshold, graph_neighbors=8)
        self.martian = MartianEngine(self.store)
        self.seeker = SeekerIndex(self.store)
        self.dreamer = Dreamer(self.store, self.martian, reflect_every=8)

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────────
    def add_memory(self, text: str, kind: str = "episodic", is_anchor: bool = False, attractor: bool = False) -> int:
        fields = {"title": "", "body": text, "snippet": "", "tags": ""}
        rid = self.store.add_room(text, kind=kind, fields=fields, is_anchor=is_anchor, attractor=attractor)
        if rid >= 0:
            # memory can optionally be indexed too (useful for Seeker if you want unified search)
            # but default: index only semantic/docs/pages/snippets
            if kind in ("semantic", "doc", "page", "snippet"):
                self.seeker.index_room(rid)
            self._enforce_capacity()
        return rid

    def add_page_result(
        self,
        title: str,
        snippet: str = "",
        body: str = "",
        url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        kind: str = "page",
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> int:
        tags_text = " ".join(tags or [])
        canonical = "\n".join([t for t in [title, snippet, body, tags_text] if t]).strip()
        fields = {"title": title or "", "snippet": snippet or "", "body": body or "", "tags": tags_text}

        rid = self.store.add_room(
            canonical,
            kind=kind,
            fields=fields,
            metadata={"url": url, "source": source, "doc_id": doc_id},
            is_anchor=False,
            attractor=False
        )
        if rid >= 0:
            self.seeker.index_room(rid)
            self._enforce_capacity()
        return rid

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval
    # ──────────────────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10, hops: int = 2, diversify: bool = True) -> List[Dict]:
        """
        Seeker-mode retrieval:
          base = fielded BM25 + phrase/bigram
          expansion = Dijkstra geodesic over store.graph
          diversification = MMR using pseudo_sim
        """
        base_scores = self.seeker.score_candidates(query, add_sim_rerank=True)
        if not base_scores:
            return []

        # seeds are top base results
        seeds = [rid for rid, _ in sorted(base_scores.items(), key=lambda x: x[1], reverse=True)[:max(6, top_k)]]

        # geodesic expand over Lotus graph
        geo_costs = self._geodesic_expand(seeds, max_hops=hops, expand_limit=90)

        combined = dict(base_scores)
        seed_peak = max((base_scores.get(s, 0.0) for s in seeds), default=0.0)

        for rid, cost in geo_costs.items():
            r = self.store.room_by_id(rid)
            if not r or r["meta"].get("archived"):
                continue
            proximity = 1.0 / (1.0 + cost)
            bonus = 0.12 * proximity
            combined[rid] = max(combined.get(rid, 0.0), 0.35 * seed_peak + bonus)

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        pool = ranked[:min(40, len(ranked))]

        if diversify:
            picked = self._mmr_select(pool, top_k=top_k, lam=0.72)
        else:
            picked = [rid for rid, _ in pool[:top_k]]

        out = []
        for rid in picked:
            rr = self.store.room_by_id(rid)
            if rr:
                out.append(rr)
                self.store.access_order.append(rid)
        return out

    def recall(self, query: str, top_k: int = 6) -> List[Dict]:
        """Martian-mode retrieval (continuity / memory recall)."""
        return self.martian.retrieve(query, top_k=top_k, min_sim=0.20, expand_hops=1)

    # ──────────────────────────────────────────────────────────────────────────
    # Dream + consolidation
    # ──────────────────────────────────────────────────────────────────────────
    def tick(self) -> Dict:
        """
        One dream tick:
          - rehearsal
          - occasional reflect
          - drift check is not automatic here (call talos_check on ingest if desired)
        """
        hub = self.dreamer.tick()
        if hub is not None:
            # newly created semantic hubs should be indexed for Seeker
            self.seeker.index_room(hub)
            self._enforce_capacity()
        return {"dream_level": self.dreamer.dream_level, "reflect_hub": hub}

    def talos_check(self, new_text: str) -> Dict:
        return self.martian.talos_check(new_text)

    def reflect(self) -> Optional[int]:
        hub = self.martian.reflect()
        if hub is not None:
            self.seeker.index_room(hub)
            self._enforce_capacity()
        return hub

    # ──────────────────────────────────────────────────────────────────────────
    # Capacity enforcement (index-safe pruning)
    # ──────────────────────────────────────────────────────────────────────────
    def _room_value(self, r: Dict) -> float:
        now = time.time()
        age_days = (now - r["meta"]["ts"]) / 86400.0
        recency = 1.0 / (1.0 + age_days)

        kind_pri = self.martian.kind_priority(r["meta"]["kind"])
        v = (
            0.40 * r["meta"]["importance"] +
            0.30 * r["meta"]["stability"] +
            0.20 * kind_pri +
            0.10 * recency
        )
        if r["meta"]["kind"] == "semantic":
            v += 0.25
        if r["meta"].get("archived"):
            v -= 0.10
        if r["id"] in self.store.anchor_ids:
            v += 1.00
        return v

    def _enforce_capacity(self):
        while len(self.store.rooms) > self.store.max_rooms:
            self._prune_one()

    def _prune_one(self):
        if not self.store.rooms:
            return

        # archive-first eviction
        candidates = [r for r in self.store.rooms if r["id"] not in self.store.anchor_ids]
        if not candidates:
            candidates = list(self.store.rooms)

        archived = [r for r in candidates if r["meta"].get("archived")]
        pool = archived if archived else candidates

        victim = min(pool, key=self._room_value)
        vid = victim["id"]

        # remove from Seeker index first
        self.seeker.remove_room(vid)

        # remove from store
        self.store.remove_room(vid)

    # ──────────────────────────────────────────────────────────────────────────
    # Geodesic expansion (Dijkstra) + MMR
    # ──────────────────────────────────────────────────────────────────────────
    def _geodesic_expand(self, seeds: List[int], max_hops: int, expand_limit: int) -> Dict[int, float]:
        best_cost: Dict[int, float] = {}
        pq: List[Tuple[float, int, int]] = []

        for s in seeds:
            best_cost[s] = 0.0
            heapq.heappush(pq, (0.0, 0, s))

        expanded = 0
        while pq and expanded < expand_limit:
            cost, hops, node = heapq.heappop(pq)
            if cost > best_cost.get(node, float("inf")) + 1e-12:
                continue
            if hops > max_hops:
                continue

            expanded += 1
            for nb, edge_cost in self.store.graph.get(node, {}).items():
                nhops = hops + 1
                if nhops > max_hops:
                    continue
                ncost = cost + edge_cost
                if ncost < best_cost.get(nb, float("inf")) - 1e-12:
                    best_cost[nb] = ncost
                    heapq.heappush(pq, (ncost, nhops, nb))

        return best_cost

    def _mmr_select(self, ranked: List[Tuple[int, float]], top_k: int, lam: float) -> List[int]:
        if not ranked:
            return []
        pool_ids = [rid for rid, _ in ranked]
        texts = {}
        for rid in pool_ids:
            r = self.store.room_by_id(rid)
            if r:
                texts[rid] = r["canonical"]
        rel = {rid: score for rid, score in ranked}

        selected = [ranked[0][0]]
        while len(selected) < min(top_k, len(ranked)):
            best_id = None
            best_mmr = -float("inf")

            for rid, _ in ranked:
                if rid in selected:
                    continue
                rt = texts.get(rid, "")
                max_sim = 0.0
                for sid in selected:
                    st = texts.get(sid, "")
                    max_sim = max(max_sim, self.store.pseudo_sim(rt, st))
                if max_sim >= 0.85:
                    max_sim = min(1.0, max_sim + 0.10)

                mmr = lam * rel.get(rid, 0.0) - (1.0 - lam) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = rid

            if best_id is None:
                break
            selected.append(best_id)

        return selected

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────────
    def status(self) -> str:
        return self.store.status()


# =============================================================================
#  Demo (single run)
# =============================================================================

if __name__ == "__main__":
    cs = CognitoSynthetica(max_rooms=120, sim_threshold=0.25)

    # Identity anchor + attractor (Martian continuity layer)
    cs.add_memory("Julian in Charlotte NC building Cognito Synthetica: memory + search + dream consolidation.", kind="commitment", is_anchor=True)
    cs.add_memory("Maintain phase-stable creative momentum; ship tangible artifacts.", kind="state", attractor=True)

    # Add some episodic fragments
    for i in range(14):
        cs.add_memory(f"Episodic fragment {i}: chaotic creative energy, planning, shifting focus.", kind="episodic")

    # Add "page results" (Seeker layer)
    pages = [
        ("IR definition",
         "What is information retrieval?",
         "Information retrieval is the process of obtaining information system resources relevant to an information need.",
         "https://example.com/ir", ["ir","definition"]),
        ("BM25 explained",
         "BM25 is a probabilistic ranking model.",
         "BM25 improves TF-IDF with term saturation and document length normalization.",
         "https://example.com/bm25", ["bm25","ranking"]),
        ("Vector space model",
         "Vector space models represent docs as vectors.",
         "Vector space models use cosine similarity and term weighting for similarity search.",
         "https://example.com/vsm", ["vectors","similarity"]),
        ("Graph-based IR",
         "Graphs connect related documents for better recall.",
         "Graph-based IR uses link analysis or similarity edges; expansion improves recall.",
         "https://example.com/graph-ir", ["graph","recall"]),
        ("Novelty & diversification",
         "Avoid redundant results.",
         "Novelty detection and result diversification reduce redundancy in search results.",
         "https://example.com/novelty", ["novelty","diversity"]),
        ("Summarization as hubs",
         "Condense long docs to short overviews.",
         "Summarization consolidates clusters into short previews and semantic hubs.",
         "https://example.com/summarize", ["summarization","hub"]),
    ]
    for title, snip, body, url, tags in pages:
        cs.add_page_result(title=title, snippet=snip, body=body, url=url, tags=tags, kind="page")

    print(cs.status())

    # Dream ticks
    for _ in range(10):
        evt = cs.tick()
        if evt["reflect_hub"] is not None:
            print(f"[Dreamer] reflect hub created: {evt['reflect_hub']}")

    print("\nStatus after dreaming:")
    print(cs.status())

    # Seeker search
    q = '"vector space" bm25 ranking search'
    print("\nSeeker search:", q)
    results = cs.search(q, top_k=6, hops=2, diversify=True)
    for i, r in enumerate(results, 1):
        title = r.get("fields", {}).get("title", "")[:70]
        url = r.get("meta", {}).get("url", None)
        print(f"{i:02d}. {title} | kind={r['meta']['kind']} | url={url} | archived={r['meta'].get('archived')}")

    # Martian recall
    print("\nMartian recall: 'creative momentum'")
    mrec = cs.recall("creative momentum", top_k=6)
    for i, r in enumerate(mrec, 1):
        print(f"{i:02d}. [{r['meta']['kind']}] {r['canonical'][:80]}...")

    # Talos drift check on a new thought
    print("\nTalos check:")
    print(cs.talos_check("I want the system to prevent rumination loops and preserve momentum."))
