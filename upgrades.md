# Dependency Feature Upgrade Assessment

Generated on 2026-05-31 after inspecting the repository dependency declarations, import usage, and the local Python 3.13 environment. The main inspection commands were:

- `uv python install 3.13 && rm -rf .venv && uv venv --python 3.13 .venv`
- `uv pip install --python .venv/bin/python -e ".[dev,test]"`
- `uv pip install --python .venv/bin/python -e ".[ner,arrow,hub,tika]"`
- AST import scan over `lexnlp/` and `scripts/`
- Runtime signature/type probe with `inspect.signature` and `importlib.metadata.version`

Important environment finding: the existing Python 3.14 `.venv` could not build `gensim==4.4.0` from source because generated C extensions still referenced CPython internals removed in Python 3.14. Recreating the environment with Python 3.13.5, which matches the repository recommendation, installed the project and all declared extras successfully.

## Installed versions observed

| Area | Package | Version observed | Primary LexNLP usage |
|---|---:|---:|---|
| HTML parsing | `beautifulsoup4` | 4.14.3 | HTML-to-text cleanup and clause extraction |
| HTML parser backend | `lxml` | 6.1.1 | Fast BeautifulSoup parser backend |
| Serialization | `cloudpickle` | 3.1.2 | Legacy model loading/dumping |
| Date parsing | `dateparser` | 1.4.0 | Multilingual fuzzy date extraction |
| Date parsing | `python-dateutil` | 2.9.0.post0 | Deterministic date parsing/relativedelta utilities |
| Search upload | `elasticsearch` | 9.4.1 | Benchmark indexing helper |
| Embeddings | `gensim` | 4.4.0 | Doc2Vec contract models |
| Persistence/cache | `joblib` | 1.5.3 | Caching, sklearn artifact loading, parallel transforms |
| NLP | `nltk` | 3.9.4 | Tokenization, POS tagging, chunking, Stanford wrappers |
| Number words | `num2words` | 0.5.14 | EN/DE amount/date word generation |
| Numeric arrays | `numpy` | 2.4.6 | Vector math, feature matrices, typed arrays |
| Tabular data | `pandas` | 2.3.3 | Batch extractor outputs and CSV/catalog flows |
| Memory/process | `psutil` | 7.2.2 | Memory-aware helpers/tests |
| Country metadata | `pycountry` | 26.2.16 | Country and subdivision extraction |
| Regex engine | `regex` | 2026.5.9 | Unicode/fuzzy/legal citation patterns |
| Reporter data | `reporters-db` | 3.2.65 | U.S. citation reporter names and variations |
| HTTP | `requests` | 2.34.2 | Catalog/model/title downloads |
| ML | `scikit-learn` | 1.8.0 | Pipelines, classifiers, vectorizers, config toggles |
| Sparse matrices | `scipy` | 1.17.1 | Sparse stacking in sklearn transformers |
| Safe model IO | `skops` | 0.14.0 | Safer sklearn persistence and model cards |
| Progress | `tqdm` | 4.67.3 | Downloads and batch extraction progress |
| Transliteration | `Unidecode` | 1.4.0 | Definition normalization |
| U.S. states | `us` | 3.2.0 | State/territory aliases |
| German numerals | `zahlwort2num` | 1.0.1 | German date/amount parsing |
| Optional Arrow | `pyarrow` | 24.0.0 | Arrow-backed pandas output |
| Optional NER | `spacy` | 3.8.14 | Opt-in token sequence classifier backend |
| Optional hub | `huggingface_hub` | 1.17.0 | Model/catalog mirror downloads |
| Optional Tika | `tika` | 3.1.0 | Document text extraction bridge |

## 100 feature-by-feature findings

1. **BeautifulSoup parser selection** — LexNLP already uses `BeautifulSoup(markup, features=...)`; keep defaulting to `lxml` because bs4's constructor accepts explicit parser features and this makes malformed legal HTML cleanup deterministic. **Status: adopted; keep.**
2. **BeautifulSoup CSS selectors** — `soup.select()` is used for clause extraction; bs4 4.14 keeps soupsieve selector support strong enough for `p, li, h1...` extraction without adding a separate DOM walker. **Status: adopted; expand selectors only by fixture.**
3. **BeautifulSoup destructive cleanup** — `tag.decompose()` fits the current script/style removal path and avoids leaving text behind; no replacement needed. **Status: adopted.**
4. **BeautifulSoup text joining** — `get_text(separator="\n", strip=True)` maps directly to downstream sentence/paragraph extractors; keep this instead of manual `.strings` flattening. **Status: adopted.**
5. **lxml parser backend** — `lxml.etree.HTMLParser` is available as a compiled extension through lxml 6.1.1; using it through bs4 is lower-maintenance than direct tree traversal for the current cleaner. **Status: adopted through bs4.**
6. **lxml streaming parse** — `lxml.etree.iterparse` is available and should be considered for very large HTML/XML filings where loading full BeautifulSoup trees is memory-heavy. **Status: candidate for large-document extractor.**
7. **lxml recovery behavior** — lxml remains the best backend for broken Office-exported HTML; keep fallback to stdlib parser only as an import-time safety net. **Status: adopted.**
8. **cloudpickle legacy load** — `cloudpickle.load(file, *, fix_imports=True, encoding=...)` remains compatible with old callable-heavy artifacts; retain only for trusted legacy assets. **Status: keep as compatibility path.**
9. **cloudpickle dump buffer callback** — `cloudpickle.dump(obj, file, protocol=None, buffer_callback=None)` supports protocol-5 buffers, but LexNLP should not generate new cloudpickle model artifacts because skops is safer. **Status: do not expand.**
10. **cloudpickle custom classes** — CloudPickler can persist dynamic classes/functions, useful for old pipeline assets; document that this is unsafe for untrusted files. **Status: legacy only.**
11. **skops safe load** — `skops.io.load(file, trusted=...)` gives explicit trust boundaries and should stay the canonical sklearn load path for new artifacts. **Status: adopted.**
12. **skops untrusted type audit** — `skops.io.get_untrusted_types(file=...)` should be run in quality gates for new model artifacts before publishing. **Status: candidate for CI hardening.**
13. **skops dump compression knobs** — `skops.io.dump(obj, file, compression=..., compresslevel=...)` can reduce model artifact size without returning to pickle. **Status: candidate for release scripts.**
14. **skops model cards** — `skops.card.Card(model, model_format=...)` is available and already aligns with LexNLP's model-card direction; use it for contract and contract-type artifacts. **Status: adopted/expand.**
15. **joblib Memory cache** — `joblib.Memory(location=..., mmap_mode=..., compress=...)` suits deterministic expensive catalog/model computations; keep using it rather than hand-written cache files. **Status: adopted.**
16. **joblib Parallel return modes** — `joblib.Parallel(..., return_as='list')` is available; streaming return modes should be evaluated for very large batch extraction jobs. **Status: candidate.**
17. **joblib delayed transforms** — Existing sklearn transformer parallelization remains appropriate; prefer joblib over multiprocessing primitives for sklearn-compatible workloads. **Status: adopted.**
18. **joblib mmap load** — `joblib.load(filename, mmap_mode=...)` can reduce memory pressure for large numpy arrays in artifacts, but only for trusted local artifacts. **Status: candidate for bundled arrays.**
19. **dateparser explicit languages/locales** — `dateparser.parse(..., languages=..., locales=..., settings=...)` is available; pass locale hints whenever extractor locale is known to avoid false positives. **Status: expand.**
20. **dateparser search_dates** — LexNLP uses `dateparser.search.search_dates`; retain it for fuzzy spans but wrap with language/date-order settings per locale. **Status: adopted/needs stricter settings.**
21. **dateparser translation data** — ES/PT modules import date translation data; continue using package-owned locale dictionaries instead of maintaining duplicate month lists. **Status: adopted.**
22. **dateparser custom language detector** — `detect_languages_function` exists on `parse`; avoid it in deterministic extractors unless backed by tests because it can add non-determinism. **Status: avoid for core extraction.**
23. **python-dateutil parserinfo** — `dateutil.parser.parse(timestr, parserinfo=..., **kwargs)` remains useful for deterministic parser variants where dateparser is too fuzzy. **Status: candidate for strict paths.**
24. **dateutil relativedelta** — Use `relativedelta` for contract duration arithmetic instead of manual month/year math. **Status: candidate audit for duration modules.**
25. **dateutil timezone handling** — Keep timezone parsing out of legal-date extraction unless the text domain requires it; date-only contract dates should remain date-like. **Status: conservative.**
26. **regex Unicode engine** — The third-party `regex` module supports the Unicode-heavy multilingual patterns LexNLP needs better than stdlib `re`. **Status: adopted.**
27. **regex fuzzy BESTMATCH** — `regex.BESTMATCH` is available; use only for bounded fuzzy legal identifiers/citations because fuzzy matching can increase false positives. **Status: selective candidate.**
28. **regex VERSION1 semantics** — `regex.VERSION1` is available for more predictable set/Unicode behavior; consider it for newly added complex patterns. **Status: candidate.**
29. **regex compiled pattern types** — Many modules compile patterns at import time; keep this for high-volume extractors and avoid recompiling per call. **Status: adopted.**
30. **regex named captures** — Prefer named groups in new extractors to reduce tuple-position coupling and simplify batch output. **Status: style recommendation.**
31. **nltk TreebankWordTokenizer** — Treebank tokenization is already used and remains a stable default for English legal text without requiring spaCy models. **Status: adopted.**
32. **nltk `word_tokenize` language parameter** — `word_tokenize(text, language='english', preserve_line=False)` is available; pass `preserve_line=True` where sentence segmentation is already handled. **Status: candidate.**
33. **nltk POS tagging** — `pos_tag(tokens, tagset=None, lang='eng')` is current; keep corpus bootstrap explicit for deterministic installs. **Status: adopted.**
34. **nltk chunking** — `RegexpParser(grammar, root_label='S', loop=1, trace=0)` remains appropriate for company/entity noun-phrase grammars. **Status: adopted.**
35. **nltk named entity chunking** — `ne_chunk(tagged_tokens, binary=False)` remains available as default NER fallback, but should be hidden behind asset checks. **Status: adopted with assets.**
36. **nltk data path cataloging** — NLTK data path management in the catalog should remain centralized; avoid per-module downloads. **Status: adopted.**
37. **spaCy model loading** — `spacy.load(name, disable=..., enable=..., exclude=...)` is available for opt-in NER; keep it optional to avoid hard model downloads. **Status: optional adopted.**
38. **spaCy blank pipelines** — `spacy.blank(lang)` can provide deterministic tokenization tests without downloading `en_core_web_sm`. **Status: candidate for unit tests.**
39. **spaCy pipeline disabling** — Use `disable`/`exclude` to load only required components for token-sequence classifiers and reduce startup latency. **Status: candidate.**
40. **spaCy listener replacement** — spaCy 3.8 supports modern pipeline component replacement patterns; use only if retraining classifier pipelines. **Status: future.**
41. **gensim Doc2Vec inference** — `Doc2Vec` remains the core interface for contract-type embeddings; keep model loading behind compatibility wrappers. **Status: adopted.**
42. **gensim TaggedDocument** — `TaggedDocument(words, tags)` is available for any future retraining scripts; prefer it to ad-hoc tuple formats. **Status: candidate for training code.**
43. **gensim callbacks** — `CallbackAny2Vec` is imported; use callbacks for progress/quality telemetry instead of print loops in training scripts. **Status: candidate.**
44. **gensim Python 3.14 risk** — Gensim 4.4.0 installed cleanly on Python 3.13 but failed to build on Python 3.14 in this environment; keep CI pinned to 3.13 until wheels/source support catches up. **Status: blocker noted.**
45. **numpy typed arrays** — `numpy.typing.NDArray` is already used; continue adding annotations around vector/math helpers. **Status: adopted.**
46. **numpy array copy semantics** — `numpy.array(..., copy=True, ndmax=...)` has explicit copy semantics in 2.x; audit any code assuming implicit no-copy behavior. **Status: candidate audit.**
47. **numpy vectorized dot/norm** — Keep cosine/vector operations vectorized with NumPy rather than Python loops. **Status: adopted.**
48. **numpy `einsum` optimization** — `numpy.einsum(..., optimize=...)` is available for dense feature math; consider only after profiling. **Status: performance candidate.**
49. **numpy NaN handling** — Modern NumPy plus sklearn HistGradientBoosting can avoid manual imputation in some classifiers. **Status: candidate with tests.**
50. **scipy sparse vstack** — `scipy.sparse.vstack(blocks, format=None, dtype=None)` maps to current sparse feature concatenation; keep it for memory efficiency. **Status: adopted.**
51. **scipy issparse** — `issparse(x)` should guard code paths that branch between dense pandas/numpy and sparse matrices. **Status: adopted.**
52. **scipy sparse format selection** — Explicitly choose CSR/CSC in new transformer outputs where downstream sklearn estimators require it. **Status: candidate.**
53. **scikit-learn Pipeline transform_input** — `Pipeline(steps, *, transform_input=None, memory=None, verbose=False)` now exposes metadata routing support; consider for sample weights/groups in model training. **Status: candidate.**
54. **scikit-learn metadata routing** — `set_config(enable_metadata_routing=True)` is available and should replace private pipeline plumbing. **Status: adopted helper; expand usage.**
55. **scikit-learn pandas output** — `set_config(transform_output='pandas')` and estimator `set_output` make feature debugging easier. **Status: adopted helper.**
56. **HistGradientBoostingClassifier** — Modern histogram gradient boosting is available and better fits NaN-tolerant tabular legal features than legacy gradient boosting. **Status: adopted helper/expand.**
57. **RandomForestClassifier defaults** — `RandomForestClassifier(n_estimators=100, max_features='sqrt', ...)` remains a stable baseline but should not be preferred over histogram boosting without benchmark evidence. **Status: keep as baseline.**
58. **TfidfVectorizer controls** — `TfidfVectorizer(tokenizer=..., analyzer=..., token_pattern=...)` aligns with legal phrase features; keep explicit tokenization to avoid default token-pattern surprises. **Status: adopted.**
59. **CountVectorizer vocabulary** — CountVectorizer is still appropriate for interpretable clause/category features where TF-IDF weighting hurts explainability. **Status: candidate by model.**
60. **NotFittedError checks** — `check_is_fitted` and `NotFittedError` remain better than catching generic exceptions in predictors. **Status: adopted.**
61. **sklearn tree internals risk** — Imports from `sklearn.tree._tree` are private and may break across versions; isolate them behind model-IO compatibility code only. **Status: risk contained.**
62. **pandas Copy-on-Write** — Pandas 2.3 supports opt-in CoW; LexNLP has helpers to prepare for pandas 3 behavior and should call them in batch scripts. **Status: adopted helper; expand.**
63. **pandas future string inference** — `pd.options.future.infer_string=True` reduces object dtype drift for text outputs. **Status: adopted helper.**
64. **pandas Arrow dtype backend** — `read_csv(..., dtype_backend='pyarrow')` is available when pyarrow is installed and suits catalog CSV reads. **Status: optional adopted.**
65. **pandas DataFrame outputs** — Batch extractors should preserve explicit column names/types and avoid returning untyped object blobs. **Status: ongoing.**
66. **pandas convert_dtypes** — `frame.convert_dtypes(dtype_backend='pyarrow')` is a low-friction post-processing step for extractor outputs. **Status: adopted helper.**
67. **pyarrow Table** — `pyarrow.Table` is available for zero-copy interchange; evaluate for high-volume batch extraction exports. **Status: candidate.**
68. **pyarrow CSV module** — Import `pyarrow.csv` directly rather than expecting `pyarrow.csv` as a top-level attribute; useful if replacing pandas CSV reads. **Status: candidate with import caveat.**
69. **pyarrow string storage** — Arrow-backed strings can lower memory use for repeated legal entity columns. **Status: candidate for batch output.**
70. **pyarrow optional extra boundary** — Keep Arrow paths optional and with graceful fallback, because core library users should not need the large pyarrow wheel. **Status: adopted.**
71. **requests Session** — `requests.Session()` remains the right abstraction for catalog downloads with shared adapters/retry config. **Status: adopted.**
72. **requests HTTPAdapter** — Existing retry adapter use is appropriate; avoid raw `requests.get` in new downloader code. **Status: adopted guideline.**
73. **requests CaseInsensitiveDict** — Keep using requests' own header container for response/header tests instead of mocks. **Status: adopted.**
74. **requests typed get** — `requests.get(url, params=None, **kwargs)` exists but should be reserved for tiny scripts/tests; production paths should take a Session. **Status: limit.**
75. **huggingface_hub `hf_hub_download`** — Single model-file fetches should use `hf_hub_download(repo_id, filename, revision=..., cache_dir=...)` for deterministic cached artifacts. **Status: optional candidate.**
76. **huggingface_hub `snapshot_download`** — Whole model/catalog mirrors should use `snapshot_download(..., revision=...)` to pin exact revisions. **Status: optional candidate.**
77. **huggingface cache/local_dir** — Prefer explicit cache/local directories so LexNLP model paths remain reproducible in CI and offline deployments. **Status: candidate.**
78. **huggingface optional dependency** — Keep hub integration optional because many deployments use GitHub release assets instead. **Status: adopted boundary.**
79. **elasticsearch client 9.x constructor** — `Elasticsearch(hosts=..., cloud_id=..., api_key=...)` has a modern keyword-only constructor; update benchmark upload docs if used externally. **Status: candidate docs.**
80. **elasticsearch helpers.bulk** — `helpers.bulk` is appropriate for benchmark uploads; keep batching out of core extraction. **Status: utility only.**
81. **elasticsearch compatibility** — Because runtime dependency now resolves to 9.4.1 while `pyproject` allows `>=8.15`, test upload helpers against both 8.x and 9.x if they become CI-critical. **Status: compatibility risk.**
82. **tqdm download progress** — `tqdm(iterable, *args, **kwargs)` and `tqdm.auto` are already used; keep auto for notebook/terminal portability. **Status: adopted.**
83. **tqdm batch extraction wrapper** — Progress should remain opt-in for library calls and default-on only in CLIs/scripts. **Status: adopted principle.**
84. **tqdm byte progress** — Model download progress bars should set byte units and totals from response headers when available. **Status: candidate.**
85. **num2words cardinal/ordinal modes** — `num2words(number, ordinal=False, lang='en', to='cardinal', **kwargs)` supports both cardinal and ordinal rendering; use it to generate amount/date variants instead of static lists. **Status: adopted/expand.**
86. **num2words locale support** — Keep locale-specific usage explicit (`lang='de'`, `lang='en'`) to avoid default-English leakage. **Status: adopted.**
87. **zahlwort2num convert** — `zahlwort2num.convert(number: str)` fits German written-number parsing and should remain isolated to German modules. **Status: adopted.**
88. **zahlwort2num error handling** — Wrap only parsing boundaries, not imports, and add fixtures for unsupported German compounds before broadening use. **Status: candidate tests.**
89. **pycountry countries registry** — `pycountry.countries` should remain the authoritative source for ISO country names/codes. **Status: adopted.**
90. **pycountry subdivisions registry** — `pycountry.subdivisions` can strengthen state/province extraction beyond U.S.-only `us` data. **Status: candidate for international entities.**
91. **pycountry fuzzy lookup** — Evaluate fuzzy country lookup only behind strict confidence tests to avoid legal-party false positives. **Status: cautious candidate.**
92. **us state objects** — `us.STATES`/`STATES_AND_TERRITORIES` remain useful for U.S. jurisdiction extraction. **Status: adopted.**
93. **us lookup helpers** — Prefer package lookup helpers/objects over hand-maintained abbreviation dictionaries where possible. **Status: candidate refactor.**
94. **reporters-db REPORTERS** — `REPORTERS` is the correct source for citation reporter metadata; do not duplicate reporter lists. **Status: adopted.**
95. **reporters-db EDITIONS** — `EDITIONS` supports edition-aware citation variation building and should stay centralized in citation modules. **Status: adopted.**
96. **reporters-db variations** — `VARIATIONS`/`VARIATIONS_ONLY` are already imported; use package data when adding citation aliases. **Status: adopted.**
97. **Unidecode transliteration** — `unidecode(string, errors='ignore', replace_str='?')` is useful for definition normalization; keep original text alongside normalized forms when spans matter. **Status: adopted with caution.**
98. **psutil Process metrics** — `psutil.Process(pid=None)` can instrument memory-heavy batch/model scripts without shelling out. **Status: candidate for benchmarks.**
99. **psutil virtual memory** — `psutil.virtual_memory()` can guard large model/vectorizer loading in CLIs and tests. **Status: candidate.**
100. **tika parser bridge** — `tika.parser.from_file` and `from_buffer` remain viable optional document text extraction paths, but should be tested with a mock/local Tika server and never required by core extractors. **Status: optional boundary.**

## Recommended next steps

1. Keep Python 3.13 as the default tested runtime until the gensim/Python 3.14 source-build issue is resolved upstream or wheels are available.
2. Add a CI/model-release check that runs `skops.io.get_untrusted_types` against new `.skops` artifacts and records the trusted type list.
3. Prefer explicit locale settings for `dateparser` calls in locale-specific extractors to reduce fuzzy-date false positives.
4. Expand use of `sklearn` metadata routing and pandas output helpers in model-training scripts where tests demonstrate unchanged predictions.
5. Keep optional heavy integrations (`pyarrow`, `spacy`, `huggingface_hub`, `tika`) behind extras and graceful fallbacks.
6. For new HTTP tests, use dependency-supported transports/sessions rather than patching global request functions.
