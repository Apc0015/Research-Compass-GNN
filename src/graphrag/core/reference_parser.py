"""Simple reference parser for common citation formats.

This module provides a lightweight ReferenceParser that extracts authors,
title, year, venue and DOI/URL from a reference string. It is intentionally
heuristic and meant to support the metadata extractor; heavy-duty parsing
can be integrated later.
"""
from typing import Dict, Optional
import re


class ReferenceParser:
    YEAR_RE = re.compile(r"(19|20)\d{2}")
    DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
    URL_RE = re.compile(r"https?://\S+", re.I)

    def parse(self, ref_text: str) -> Dict[str, Optional[str]]:
        """Parse a single reference string into components.

        Returns a dict with keys: authors, title, year, venue, doi, url
        """
        data = {
            'authors': None,
            'title': None,
            'year': None,
            'venue': None,
            'doi': None,
            'url': None,
        }

        # DOI
        m = self.DOI_RE.search(ref_text)
        if m:
            data['doi'] = m.group(0)

        # URL
        m = self.URL_RE.search(ref_text)
        if m:
            data['url'] = m.group(0)

        # Year
        m = self.YEAR_RE.search(ref_text)
        if m:
            data['year'] = m.group(0)

        # Authors (heuristic: leading segment before year or a dash)
        parts = re.split(r"\.|\)|;|\(|\[|\]|--", ref_text)
        if parts:
            lead = parts[0]
            # try splitting by colon or comma
            if ':' in lead:
                lead = lead.split(':')[0]
            auth_candidates = [a.strip() for a in re.split(r',| and ', lead) if len(a.strip()) > 2]
            if auth_candidates:
                data['authors'] = ', '.join(auth_candidates[:4])

        # Title heuristic: quoted strings or segments between year and period
        qm = re.search(r'"([^"]{10,200})"', ref_text)
        if qm:
            data['title'] = qm.group(1).strip()
        else:
            # fallback: try to find segment between year and next period
            if data['year']:
                yidx = ref_text.find(data['year'])
                if yidx != -1:
                    tail = ref_text[yidx+4:]
                    # split on '.' and take first meaningful chunk
                    parts = [p.strip() for p in tail.split('.') if len(p.strip()) > 10]
                    if parts:
                        data['title'] = parts[0]

        # Very naive venue detection: common journal/conference words
        v = re.search(r"(Proceedings of|Journal of|Transactions on|Conference on|Workshop on)\s+([A-Za-z0-9\-\s]+)", ref_text)
        if v:
            data['venue'] = (v.group(1) + ' ' + (v.group(2) or '')).strip()

        return data


__all__ = ['ReferenceParser']
