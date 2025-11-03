"""Engine to infer missing relationships using graph heuristics and optional ML.

This lightweight engine provides methods to suggest missing citations, infer
topic similarities, and propagate influence by walking citation chains.
"""
from __future__ import annotations

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class RelationshipInferenceEngine:
    """Inference helpers for relationships.

    The class expects to be given a GraphManager-like object with either a
    Neo4j driver or an in-memory networkx graph available as `_graph` and
    `_node_props`.
    """

    def __init__(self, graph):
        self.graph = graph

    def infer_topic_similarity(self, topic_ids: List[str], threshold: float = 0.5) -> List[Dict]:
        """Compute simple topic similarity based on shared paper membership.

        Returns list of dicts: {topic1, topic2, score}
        """
        pairs = []
        try:
            # Build mapping topic -> set(paper_ids)
            topic_to_papers = {}
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    for tid in topic_ids:
                        res = session.run("MATCH (t:Topic {id:$id})<-[:DISCUSSES]-(p:Paper) RETURN p.id as id", id=tid)
                        topic_to_papers[tid] = set([r['id'] for r in res])
            else:
                # use node_props
                for tid in topic_ids:
                    papers = [pid for pid, props in self.graph._node_props.items() if props.get('label') == 'Paper' and tid in (props.get('topics') or [])]
                    topic_to_papers[tid] = set(papers)

            tids = list(topic_to_papers.keys())
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a = tids[i]
                    b = tids[j]
                    sa = topic_to_papers.get(a, set())
                    sb = topic_to_papers.get(b, set())
                    if not sa or not sb:
                        continue
                    inter = sa.intersection(sb)
                    union = sa.union(sb)
                    score = len(inter) / max(len(union), 1)
                    if score >= threshold:
                        pairs.append({'topic1': a, 'topic2': b, 'score': float(score)})
        except Exception:
            logger.exception("Failed to infer topic similarity")

        return pairs

    def infer_missing_citations(self, candidate_paper_id: str, top_k: int = 10) -> List[Dict]:
        """Suggest papers that candidate_paper should cite based on topic overlap and methods.

        Returns list of dicts: {paper_id: str, score: float, reason: str}
        """
        suggestions = []
        try:
            # Very simple heuristic: find papers with high keyword overlap that are older
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    q = "MATCH (p:Paper {id:$id}) OPTIONAL MATCH (q:Paper) WHERE q.id <> $id RETURN p, q LIMIT 1000"
                    res = session.run(q, id=candidate_paper_id)
                    # fallback: empty suggestion list
            else:
                # naive: compare keywords
                cand = self.graph._node_props.get(candidate_paper_id) or {}
                cand_k = set((cand.get('keywords') or []) + (cand.get('title') or '').lower().split())
                scores = []
                for pid, props in self.graph._node_props.items():
                    if pid == candidate_paper_id:
                        continue
                    kws = set((props.get('keywords') or []) + (props.get('title') or '').lower().split())
                    overlap = len(cand_k.intersection(kws))
                    if overlap > 0:
                        scores.append((pid, float(overlap)))
                scores.sort(key=lambda x: x[1], reverse=True)
                for pid, sc in scores[:top_k]:
                    suggestions.append({'paper_id': pid, 'score': sc, 'reason': 'keyword_overlap'})
        except Exception:
            logger.exception("Failed to infer missing citations")

        return suggestions

    def propagate_influence(self, source_paper_id: str, max_depth: int = 3) -> List[Dict]:
        """Calculate indirect influence scores from source paper by walking citation graph.

        Returns list of dicts: {paper_id, path_length, weight}
        """
        results = []
        try:
            if getattr(self.graph, '_use_neo4j', False):
                with self.graph.driver.session() as session:
                    # collect reachable papers within depth
                    q = "MATCH (s:Paper {id:$id})-[r:CITES*1..$d]->(t:Paper) RETURN DISTINCT t.id as id, length((s)-[r]->(t)) as dist"
                    res = session.run(q, id=source_paper_id, d=max_depth)
                    for r in res:
                        dist = int(r['dist'])
                        weight = 1.0 / (dist + 1)
                        results.append({'paper_id': r['id'], 'path_length': dist, 'weight': weight})
            else:
                # BFS on in-memory graph considering CITES edges
                from collections import deque
                q = deque()
                q.append((source_paper_id, 0))
                seen = {source_paper_id}
                while q:
                    cur, d = q.popleft()
                    if d >= max_depth:
                        continue
                    for _, tgt, data in self.graph._graph.out_edges(cur, data=True):
                        if data.get('type') == 'CITES' and tgt not in seen:
                            seen.add(tgt)
                            weight = 1.0 / (d + 2)
                            results.append({'paper_id': tgt, 'path_length': d + 1, 'weight': weight})
                            q.append((tgt, d + 1))
        except Exception:
            logger.exception("Failed to propagate influence")

        return results
