# Mini Search Engine

A small Python search engine for a predefined set of web pages. It fetches each
target URL once, indexes the page text, builds a directed graph from links
between target URLs, and supports three ranking modes.

## Setup

Use Python 3.10 or later:

```bash
python -m pip install requests beautifulsoup4 networkx numpy scipy
```

## Usage

```python
from search_engine import SearchEngine

target_urls = [
    "https://example.com/",
    "https://example.org/",
]

# Fetches and indexes the pages, calculates PageRank, and runs the surfer
# simulation once (100,000 moves).
engine = SearchEngine(target_urls)

# The default is hybrid: normalized TF-IDF relevance combined with PageRank.
results = engine.search("search word")
for result in results:
    print(result["title"], result["url"], result["score_display"])
    print(result["snippet"])

# Alternative rankings:
old_results = engine.search("search word", ranking="old")
surfer_results = engine.search("search word", ranking="surfer")
```

## Run the website

Start the local web interface:

```bash
python app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Paste one target
URL per line, enter a query, choose a ranking, and select **Search**. The first
search for a URL list fetches and indexes the pages; later searches reuse that
index until a different URL list is submitted. The server binds to localhost
only.

`search` returns dictionaries containing `title`, `url`, `score`,
`score_display`, and `snippet`. The snippet wraps matching words in `<mark>`
tags; escape the text as appropriate before rendering it in an HTML page.

## Ranking modes

- **`old`**: counts case-insensitive, whole-word occurrences of the query terms
  in page text and sorts by count.
- **`surfer`**: performs one random-surfer simulation when the engine is
  initialized. It starts on a random target page, makes exactly 100,000 moves,
  jumps to a random page 15% of the time, and otherwise follows a random
  in-scope outgoing link. A page with no in-scope links jumps to a random page.
  Returned `score` is a percentage from 0 to 100 and `score_display` formats
  it to one decimal place. Repeated searches reuse the same simulation result.
- **`hybrid`**: ranks matching pages by normalized TF-IDF relevance blended
  with NetworkX PageRank. The default weights are 0.8 for relevance and 0.2
  for authority; pass weights to `SearchEngine` to change them (they must sum
  to 1).

The inverted index maps each non-stop-word token to the document IDs containing
it. Matching is an OR across query terms. Outgoing links are restricted to the
provided target URL list. Fetch errors are raised with the affected URL rather
than silently dropping a page.
