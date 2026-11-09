"""A small search engine for a predefined set of URLs.

The engine fetches each URL once, builds content and link indexes, then exposes
three ranking modes: occurrence count, random-surfer visits, and TF-IDF blended
with NetworkX PageRank.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit

import networkx as nx
import requests
from bs4 import BeautifulSoup


STOP_WORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm
    i've if in into is isn't it it's its itself let's me more most mustn't my
    myself no nor not of off on once only or other ought our ours ourselves
    out over own same shan't she she'd she'll she's should shouldn't so some
    such than that that's the their theirs them themselves then there there's
    these they they'd they'll they're they've this those through to too under
    until up very was wasn't we we'd we'll we're we've were weren't what
    what's when when's where where's which while who who's whom why why's
    with won't would wouldn't you you'd you'll you're you've your yours
    yourself yourselves
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?", re.IGNORECASE)
SURFER_STEPS = 100_000
DAMPING_FACTOR = 0.85


@dataclass(frozen=True)
class Document:
    """Fetched page content and the links it has to other target URLs."""

    doc_id: int
    url: str
    title: str
    text: str
    tokens: tuple[str, ...]
    outgoing_links: tuple[str, ...]


def normalize_url(url: str) -> str:
    """Normalize URL spelling for matching links against the target list."""
    without_fragment, _fragment = urldefrag(url.strip())
    parts = urlsplit(without_fragment)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    if path == "/":
        path = ""
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def tokenize(text: str) -> list[str]:
    """Lower-case and tokenize text, dropping standard English stop words."""
    return [
        match.group(0).lower()
        for match in TOKEN_RE.finditer(text)
        if match.group(0).lower() not in STOP_WORDS
    ]


class SearchEngine:
    """Fetch, index, and search a fixed collection of web pages.

    Surfer visits and PageRank are calculated once during initialization.
    Calling ``search`` repeatedly does not re-fetch pages or re-run the surfer.
    """

    def __init__(
        self,
        urls: Iterable[str],
        *,
        request_timeout: float = 10.0,
        relevance_weight: float = 0.8,
        authority_weight: float = 0.2,
    ) -> None:
        self.urls = list(dict.fromkeys(urls))
        if not self.urls:
            raise ValueError("Provide at least one target URL.")
        if not math.isclose(relevance_weight + authority_weight, 1.0):
            raise ValueError("relevance_weight and authority_weight must sum to 1.")

        self._url_by_normalized = {normalize_url(url): url for url in self.urls}
        if len(self._url_by_normalized) != len(self.urls):
            raise ValueError("Target URLs must be unique after URL normalization.")

        self.documents = self._fetch_documents(request_timeout)
        self._doc_id_by_url = {document.url: document.doc_id for document in self.documents}
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        self._term_frequencies: dict[int, Counter[str]] = {}
        self._build_content_index()

        self.graph = self._build_link_graph()
        self.pagerank = (
            nx.pagerank(self.graph, alpha=DAMPING_FACTOR)
            if self.graph.number_of_nodes()
            else {}
        )
        self.surfer_visits = self._run_random_surfer()
        self.surfer_scores = {
            doc_id: visits / SURFER_STEPS for doc_id, visits in self.surfer_visits.items()
        }

        self.relevance_weight = relevance_weight
        self.authority_weight = authority_weight

    def _fetch_documents(self, timeout: float) -> list[Document]:
        target_urls = set(self._url_by_normalized)
        documents: list[Document] = []
        headers = {"User-Agent": "MiniSearchEngine/1.0"}

        with requests.Session() as session:
            for doc_id, url in enumerate(self.urls):
                try:
                    response = session.get(url, timeout=timeout, headers=headers)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc

                soup = BeautifulSoup(response.text, "html.parser")
                for element in soup(["script", "style"]):
                    element.decompose()

                title_element = soup.title
                title = title_element.get_text(" ", strip=True) if title_element else url
                text = soup.get_text(" ", strip=True)
                links: list[str] = []
                for anchor in soup.find_all("a", href=True):
                    absolute_url = urljoin(response.url, anchor["href"])
                    normalized = normalize_url(absolute_url)
                    if normalized in target_urls:
                        links.append(self._url_by_normalized[normalized])

                documents.append(
                    Document(
                        doc_id=doc_id,
                        url=url,
                        title=title,
                        text=text,
                        tokens=tuple(tokenize(text)),
                        outgoing_links=tuple(links),
                    )
                )
        return documents

    def _build_content_index(self) -> None:
        for document in self.documents:
            frequencies = Counter(document.tokens)
            self._term_frequencies[document.doc_id] = frequencies
            for term in frequencies:
                self.inverted_index[term].add(document.doc_id)

    def _build_link_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(document.doc_id for document in self.documents)
        for document in self.documents:
            link_counts = Counter(document.outgoing_links)
            for linked_url, count in link_counts.items():
                graph.add_edge(
                    document.doc_id,
                    self._doc_id_by_url[linked_url],
                    weight=count,
                )
        return graph

    def _run_random_surfer(self) -> dict[int, int]:
        """Simulate exactly 100,000 moves and count each move's destination."""
        doc_ids = [document.doc_id for document in self.documents]
        visits = dict.fromkeys(doc_ids, 0)
        current = random.choice(doc_ids)

        for _ in range(SURFER_STEPS):
            outgoing = self.documents[current].outgoing_links
            if random.random() < 0.15 or not outgoing:
                current = random.choice(doc_ids)
            else:
                linked_url = random.choice(outgoing)
                current = self._doc_id_by_url[linked_url]
            visits[current] += 1
        return visits

    def _query_terms(self, query: str) -> list[str]:
        return tokenize(query)

    def _tf_idf_scores(self, terms: list[str], matching_ids: set[int]) -> dict[int, float]:
        document_count = len(self.documents)
        scores: dict[int, float] = {}
        for doc_id in matching_ids:
            term_counts = self._term_frequencies[doc_id]
            token_count = sum(term_counts.values())
            score = 0.0
            for term in set(terms):
                frequency = term_counts.get(term, 0)
                if frequency and token_count:
                    document_frequency = len(self.inverted_index[term])
                    inverse_document_frequency = math.log(
                        (document_count + 1) / (document_frequency + 1)
                    ) + 1
                    score += (frequency / token_count) * inverse_document_frequency
            scores[doc_id] = score
        return scores

    @staticmethod
    def _make_snippet(text: str, terms: set[str], radius: int = 80) -> str:
        match = next(
            (
                found
                for term in sorted(terms, key=len, reverse=True)
                if (found := re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE))
            ),
            None,
        )
        if match is None:
            return text[: radius * 2].strip()

        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        snippet = text[start:end].strip()
        if start:
            snippet = "…" + snippet
        if end < len(text):
            snippet += "…"

        for term in sorted(terms, key=len, reverse=True):
            snippet = re.sub(
                rf"\b({re.escape(term)})\b",
                r"<mark>\1</mark>",
                snippet,
                flags=re.IGNORECASE,
            )
        return snippet

    def search(self, query: str, *, ranking: str = "hybrid", limit: int = 10) -> list[dict]:
        """Return matching results ordered by ``old``, ``surfer``, or ``hybrid``.

        ``old`` counts case-insensitive whole-word query terms. ``surfer`` sorts
        by the one-time random-surfer simulation. ``hybrid`` combines normalized
        TF-IDF relevance and NetworkX PageRank.
        """
        if ranking not in {"old", "surfer", "hybrid"}:
            raise ValueError("ranking must be 'old', 'surfer', or 'hybrid'.")
        if limit < 0:
            raise ValueError("limit cannot be negative.")
        if not query.strip() or limit == 0:
            return []

        terms = self._query_terms(query)
        if not terms:
            return []
        matching_ids = set().union(
            *(self.inverted_index.get(term, set()) for term in set(terms))
        )
        if not matching_ids:
            return []

        tf_idf = self._tf_idf_scores(terms, matching_ids)
        max_relevance = max(tf_idf.values(), default=0.0)
        query_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(term) for term in set(terms)) + r")\b",
            re.IGNORECASE,
        )

        results: list[dict] = []
        for doc_id in matching_ids:
            document = self.documents[doc_id]
            old_score = len(query_pattern.findall(document.text))
            normalized_relevance = (
                tf_idf[doc_id] / max_relevance if max_relevance else 0.0
            )
            hybrid_score = (
                self.relevance_weight * normalized_relevance
                + self.authority_weight * self.pagerank.get(doc_id, 0.0)
            )
            if ranking == "old":
                score = float(old_score)
            elif ranking == "surfer":
                score = self.surfer_scores[doc_id] * 100
            else:
                score = hybrid_score

            results.append(
                {
                    "title": document.title,
                    "url": document.url,
                    "score": score,
                    "score_display": (
                        f"{score:.1f}%" if ranking == "surfer" else f"{score:.4f}"
                    ),
                    "snippet": self._make_snippet(document.text, set(terms)),
                    "old_score": old_score,
                    "surfer_score": self.surfer_scores[doc_id] * 100,
                    "tf_idf_score": tf_idf[doc_id],
                    "pagerank_score": self.pagerank.get(doc_id, 0.0),
                }
            )

        return sorted(results, key=lambda result: result["score"], reverse=True)[:limit]
