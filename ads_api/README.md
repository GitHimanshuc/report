# ads_api

## Goal

This directory builds and maintains a local reference database for a small set of base papers.

The main goal is to map numbered references from a paper-specific text file onto ADS bibcodes in
`database.json`, then store stable aliases such as `__sxs25_77`. That makes it possible to later
generate a BibTeX file where a paper can cite its own numbered references by alias instead of by
retyping the full entry.

## Main Files

- `database.json`
  - Main local database.
  - `base_papers` stores each source paper and its metadata.
  - A base paper may also contain `alises_generated` to mark that its numbered references have
    already been processed into aliases.
  - `joined_bibcodes` is the set of all referenced bibcodes collected across all base papers.
  - `joined_refs` stores the BibTeX for each referenced bibcode, and may also contain `aliases`.

- `generate_database.py`
  - Talks to ADS using `ADS_API_KEY`.
  - Fills in missing `bibcodes` and `bibtex` for base papers.
  - Pulls missing referenced BibTeX entries into `joined_refs`.
  - Builds per-base embedding CSVs for papers that have `reference_txt_file`.

- `embed_joined_refs.py`
  - Creates a single embedding CSV for all entries in `joined_refs`.
  - Uses cleaned BibTeX text as embedding input.
  - Mostly useful for global matching experiments.

- `find_bibcode_match.py`
  - Matches free-form reference lines against an embedding CSV.
  - Supports exact arXiv matching, embedding ranking, and numeric reranking.
  - Useful for debugging matching quality on text files.

- `add_aliases_to_database.py`
  - Main aliasing script.
  - For each base paper with `reference_txt_file`, it builds a per-paper embedding CSV using only
    that paper's own `bibcodes`.
  - It then matches each non-empty line of the text file against that restricted set.
  - On success it adds an alias like `identifier_77` to the matched `joined_refs` entry.
  - It uses the current hardcoded embedding model from the codebase, so the per-paper CSVs are tied
    to that exact setup.

- `generate_alias_bib.py`
  - Reads the final `database.json`.
  - Looks through `joined_refs` for entries that contain `aliases`.
  - Writes one BibTeX entry per alias, even if that means the same underlying BibTeX entry appears
    multiple times.
  - Each output key is named `bibcode_alias`.

- `*.txt`
  - Paper-specific numbered reference lists.
  - The first non-empty line is reference 1, the second is reference 2, and so on.
  - These line numbers are what drive aliases such as `__owen19_12`.

## Environment Variables

- `ADS_API_KEY`
  - Required by `generate_database.py`.

- `OPENROUTER_API_KEY_EMBEDDING`
  - Required by `embed_joined_refs.py`, `find_bibcode_match.py`, and `add_aliases_to_database.py`.

## How To Add a New Base Paper

1. Add the paper to `database.json` under `base_papers`.

Example:

```json
"2012Example.....123A": {
    "identifier": "__example12",
    "reference_txt_file": "./example_2012.txt"
}
```

Notes:
- `identifier` should be short and stable.
- `reference_txt_file` should point to a text file in this directory unless you have a reason to
  place it elsewhere.
- The text file should contain one reference per line, in the same order as the paper's
  bibliography.
- The text file can be created from the paper PDF using `pdftotext` or a similar extraction tool,
  then cleaned into one reference per line.

2. Create the corresponding text file.

Example:

```text
First reference...
Second reference...
Third reference...
```

3. Run:

```bash
python ads_api/generate_database.py
```

This will:
- fetch the base paper's references from ADS if needed
- update `joined_refs`
- create the per-paper embedding CSV

If the embedding model or embedding-text normalization is changed in the code, regenerate the
embedding CSVs before matching again. The current CSV reuse logic keys off `bibcode`, so old CSVs
will otherwise remain on disk and keep using the previous embedding setup.

4. Run:

```bash
python ads_api/add_aliases_to_database.py --base-bibcode 2012Example.....123A
```

This will try to match each line in the text file and write aliases into `database.json`.
If the run finishes without failures, it also sets `base_papers[...]["alises_generated"] = true`.

5. Inspect the updated `aliases` fields in `joined_refs`.

If matching looks wrong, rerun with:

```bash
python ads_api/add_aliases_to_database.py --base-bibcode 2012Example.....123A --debug --regenerate-aliases
```

6. Generate the alias-based BibTeX file:

```bash
python ads_api/generate_alias_bib.py
```

By default this writes `ads_api/aliases.bib`. Each alias becomes its own repeated BibTeX entry with
a key of the form `bibcode_alias`.

## Typical Workflow

```bash
python ads_api/generate_database.py
python ads_api/add_aliases_to_database.py
python ads_api/generate_alias_bib.py
```

Use `find_bibcode_match.py` when you want to debug matching behavior on a text file without
writing aliases.

By default, `add_aliases_to_database.py` skips base papers where `alises_generated` is already
true. Use `--regenerate-aliases` to force a fresh aliasing pass.

## Matching Paths

The matcher can currently accept a reference through one of these paths:

- `arxiv`
  - If the input contains an arXiv identifier like `1602.03837` and that token appears directly in
    one of the candidate BibTeX entries, that match is accepted immediately.

- `embedding`
  - The input text is embedded and compared against the candidate embeddings with cosine
    similarity.
  - This path is accepted only if the top embedding score is high enough and the score falls off
    enough by rank 5.

- `numeric-all-match`
  - If embedding acceptance fails, the matcher looks at the top embedding candidates and extracts
    the larger numeric tokens from the input, such as years, volumes, and article/page numbers.
  - It then does plain substring checks for those tokens in the raw BibTeX.
  - If exactly one candidate matches all of the weighted numeric tokens, and the total numeric
    weight is at least the minimum threshold, that candidate is accepted.

- `numeric-rerank`
  - If there is not a unique full numeric match, the matcher reranks the top embedding candidates
    by how much weighted numeric overlap they have with the input.
  - This path is accepted when the best normalized numeric score is strong enough and the next best
    score is low enough, so the numeric evidence separates one candidate from the rest.

Numeric reranking is only attempted when the top embedding score is at least `0.60`. If the top
embedding candidate is weaker than that, the matcher stops and raises an error instead of using the
numeric fallback.

If none of these paths is confident enough, the matcher raises an error instead of forcing a match.
