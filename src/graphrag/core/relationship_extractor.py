"""Extract relationships from academic paper text.

This module provides heuristics to find citations, method mentions, dataset usage
and infer topic relationships from paper text. It's intentionally conservative
and returns structured dicts with confidence scores so callers can decide how to
apply the relationships.
"""
from __future__ import annotations

import re
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Heuristic extractor for relationships from paper text."""

    CITATION_PATTERN = re.compile(r"\[(\d+)\]|\(([^)]+, \d{4})\)")
    YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

    def extract_citations_from_text(self, paper_text: str, reference_list: List[str]) -> List[Dict]:
        """Match in-text citations to reference list and capture context.

        Returns list of dicts: {cited_paper: str (reference text), context: str, section: str}
        """
        results = []
        try:
            # naive: find citation markers and return surrounding sentence as context
            for m in self.CITATION_PATTERN.finditer(paper_text[:200000]):
                span = m.span()
                start = max(0, span[0] - 200)
                end = min(len(paper_text), span[1] + 200)
                context = paper_text[start:end].strip()

                # Try to match numeric citation to reference list
                num = m.group(1)
                cited = None
                if num and reference_list and num.isdigit():
                    idx = int(num) - 1
                    if 0 <= idx < len(reference_list):
                        cited = reference_list[idx]

                # If author-year style, try to find best-match in reference_list
                if not cited and m.group(2):
                    # just include the string as best-effort
                    cited = m.group(2)

                if cited:
                    results.append({'cited_paper': cited, 'context': context, 'section': None})

        except Exception:
            logger.exception("Failed to extract citations from text")

        return results

    def extract_method_usage(self, paper_text: str, known_methods: List[str]) -> List[Dict]:
        """Find occurrences of known method names and classify usage.

        Returns list of dicts: {method: str, usage: 'uses'|'proposes'|'compares'|'improves', evidence: str}
        """
        results = []
        try:
            lower = paper_text.lower()
            for method in known_methods:
                if method.lower() in lower:
                    # extract small context window
                    idx = lower.find(method.lower())
                    start = max(0, idx - 150)
                    end = min(len(lower), idx + len(method) + 150)
                    context = paper_text[start:end]

                    # simple heuristics
                    usage = 'uses'
                    if re.search(r"propose|introduce|novel|we (present|propose)", context, flags=re.I):
                        usage = 'proposes'
                    elif re.search(r"compare|outperform|baseline", context, flags=re.I):
                        usage = 'compares'
                    elif re.search(r"improv|better than|gain", context, flags=re.I):
                        usage = 'improves'

                    results.append({'method': method, 'usage': usage, 'evidence': context})
        except Exception:
            logger.exception("Failed to extract method usage")

        return results

    def extract_dataset_usage(self, paper_text: str, known_datasets: List[str]) -> List[Dict]:
        """Find dataset mentions and extract simple evaluation metrics via regex.

        Returns list of dicts: {dataset: str, metrics: Dict, evidence: str}
        """
        results = []
        try:
            lower = paper_text.lower()
            for ds in known_datasets:
                if ds.lower() in lower:
                    idx = lower.find(ds.lower())
                    start = max(0, idx - 200)
                    end = min(len(lower), idx + len(ds) + 200)
                    context = paper_text[start:end]

                    # try to extract simple metrics like 'accuracy: 90%' or 'F1: 0.82'
                    metrics = {}
                    acc = re.search(r"accuracy[:\s]+(\d{1,3}(?:\.\d+)?%)", context, flags=re.I)
                    if acc:
                        metrics['accuracy'] = acc.group(1)
                    f1 = re.search(r"f1[:\s]+(0?\.\d+|1(?:\.0)?)", context, flags=re.I)
                    if f1:
                        metrics['f1'] = float(f1.group(1))

                    results.append({'dataset': ds, 'metrics': metrics, 'evidence': context})
        except Exception:
            logger.exception("Failed to extract dataset usage")

        return results

    def infer_topic_relationships(self, paper: "PaperNode", known_topics: List["TopicNode"]) -> List[Dict]:
        """Match paper content to known topics with confidence.

        Uses keyword overlap and simple score; callers may augment with embeddings.
        Returns list of dicts: {topic_id, confidence, evidence}
        """
        results = []
        try:
            text = ' '.join(filter(None, [paper.title or '', paper.abstract or ''])).lower()
            for t in known_topics:
                name = (t.name or '').lower()
                if not name:
                    continue
                # compute simple overlap score
                words = set(re.findall(r"\w+", name))
                if not words:
                    continue
                matches = sum(1 for w in words if w in text)
                confidence = matches / max(len(words), 1)
                if confidence > 0:
                    results.append({'topic_id': t.id, 'confidence': float(confidence), 'evidence': name})
        except Exception:
            logger.exception("Failed to infer topic relationships")

        return results

    def detect_method_innovations(self, paper_text: str) -> List[Dict]:
        """Identify if paper proposes a new method; return method name and description.

        Very heuristic: looks for patterns like 'we propose X' or 'we introduce X'.
        """
        results = []
        try:
            for m in re.finditer(r"we (?:propose|introduce|present) (?:the )?([A-Z][A-Za-z0-9_-]{2,})", paper_text):
                method = m.group(1)
                start = max(0, m.start() - 100)
                end = min(len(paper_text), m.end() + 300)
                desc = paper_text[start:end]
                results.append({'method': method, 'description': desc})
        except Exception:
            logger.exception("Failed to detect method innovations")

        return results
    
    def extract_relationships(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all types of relationships from text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of relationship types and their extracted relationships
        """
        relationships = {
            'cites': [],
            'authored_by': [],
            'published_in': [],
            'discusses': [],
            'uses_method': [],
            'evaluated_on': []
        }
        
        try:
            # Extract citations
            citations = self.extract_citations_from_text(text, [])
            for citation in citations:
                relationships['cites'].append({
                    'source': 'current_paper',
                    'target': citation.get('cited_paper', 'unknown'),
                    'type': 'citation',
                    'context': citation.get('context', ''),
                    'confidence': 0.8
                })
            
            # Extract method usage
            methods = self.extract_method_usage(text, ['machine learning', 'deep learning', 'neural network'])
            for method in methods:
                relationships['uses_method'].append({
                    'source': 'current_paper',
                    'target': method.get('method', 'unknown'),
                    'type': 'method_usage',
                    'usage': method.get('usage', 'uses'),
                    'evidence': method.get('evidence', ''),
                    'confidence': 0.7
                })
            
            # Extract dataset usage
            datasets = self.extract_dataset_usage(text, ['dataset', 'benchmark', 'evaluation'])
            for dataset in datasets:
                relationships['evaluated_on'].append({
                    'source': 'current_paper',
                    'target': dataset.get('dataset', 'unknown'),
                    'type': 'evaluation',
                    'metrics': dataset.get('metrics', {}),
                    'evidence': dataset.get('evidence', ''),
                    'confidence': 0.6
                })
            
            # Add generic authorship relationship
            relationships['authored_by'].append({
                'source': 'current_paper',
                'target': 'author_unknown',
                'type': 'authorship',
                'confidence': 0.5
            })
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
        
        return relationships
