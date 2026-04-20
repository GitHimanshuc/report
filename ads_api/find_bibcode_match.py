import csv
import json
import math
import os
import re
from pathlib import Path

try:
    from embed_joined_refs import default_output_path, fetch_embeddings
except ModuleNotFoundError:
    from ads_api.embed_joined_refs import default_output_path, fetch_embeddings


# INPUT_TEXT = """Abbott B P et al. (LIGO Scientific, Virgo) 2016 Phys. Rev. Lett. 116 061102 [arXiv:1602.03837]"""
INPUT_TEXT = """T. W. Baumgarte and S. L. Shapiro, Numerical Relativity: Solving Einstein's Equations on the Computer (Cambridge University Press, New York, 2010)."""
# INPUT_TEXT = """C. O. Lousto and Y. Zlochower (2013), 1312.5775."""
# INPUT_TEXT = """T. Futamase and Y. Itoh, Living Rev. Rel. 10, 2 (2007)."""
# INPUT_TEXT = """R. Owen, Phys. Rev. D 80, 084012 (2009)."""
INPUT_TEXT = """J. Blackman, S. E. Field, M. A. Scheel, C. R. Galley, C. D. Ott, M. Boyle, L. E. Kidder, H. P. Pfeiffer, and B. Szilagyi, Phys. Rev. D 96, 024058 (2017)"""

MODEL = "openai/text-embedding-3-small"
TOP_K_FALLBACK = 5
DEBUG = False
EMBEDDING_ACCEPT_SCORE = 0.75
EMBEDDING_ACCEPT_FALLOFF = 0.15
NUMERIC_ACCEPT_SCORE = 0.8
NUMERIC_REJECT_NEXT_SCORE = 0.5
SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = SCRIPT_DIR / "database.json"
EMBEDDINGS_CSV = default_output_path(SCRIPT_DIR, MODEL)


def load_database(database_path: Path) -> dict:
    with database_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_embeddings(csv_path: Path) -> list[tuple[str, list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or not header or header[0] != "bibcode":
            raise ValueError(f"Unexpected CSV format in {csv_path}")

        for row in reader:
            bibcode = row[0]
            embedding = [float(value) for value in row[1:]]
            rows.append((bibcode, embedding))

    if not rows:
        raise ValueError(f"No embeddings found in {csv_path}")

    return rows


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Encountered a zero-length embedding vector")
    return dot / (norm_a * norm_b)


def extract_arxiv_id(text: str) -> str | None:
    match = re.search(r"\b\d{4}\.\d{4,5}\b", text)
    if match is None:
        return None
    return match.group(0)


def find_arxiv_match(database: dict, arxiv_id: str) -> tuple[str, dict] | None:
    for bibcode, entry in database["joined_refs"].items():
        bibtex = entry.get("bibtex", "")
        if isinstance(bibtex, str) and arxiv_id in bibtex:
            return bibcode, entry
    return None


def extract_numeric_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\d+\.\d+|\d+", text)
    unique_tokens: list[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)
    return unique_tokens


def numeric_token_weight(token: str) -> int:
    return sum(character.isdigit() for character in token)


def embedding_falloff(scored_matches: list[tuple[float, str]], rank: int = 5) -> float:
    if not scored_matches:
        return 0.0
    end_index = min(rank, len(scored_matches)) - 1
    return scored_matches[0][0] - scored_matches[end_index][0]


def print_top10_decay(scored_matches: list[tuple[float, str]]) -> None:
    scores = [score for score, _ in scored_matches]
    top_scores = scores[:10]

    print("\nTop 10 scores and dropoff")
    for index, score in enumerate(top_scores, start=1):
        if index == 1:
            print(f"{index:2d}. {score:.6f}")
            continue

        previous_score = top_scores[index - 2]
        absolute_drop = previous_score - score
        relative_drop = 0.0 if previous_score == 0.0 else absolute_drop / previous_score
        print(
            f"{index:2d}. {score:.6f}  "
            f"drop_from_prev={absolute_drop:.6f}  "
            f"relative_drop={relative_drop:.6%}"
        )


def rerank_top_matches_by_numeric_overlap(
    scored_matches: list[tuple[float, str]],
    database: dict,
    input_text: str,
) -> list[tuple[float, float, int, list[str], str]]:
    numeric_tokens = extract_numeric_tokens(input_text)
    top_candidates = scored_matches[:10]

    if not numeric_tokens:
        return [(score, 0.0, 0, [], bibcode) for score, bibcode in top_candidates]

    total_numeric_weight = sum(numeric_token_weight(token) for token in numeric_tokens)
    if total_numeric_weight == 0:
        return [(score, 0.0, 0, [], bibcode) for score, bibcode in top_candidates]

    reranked_matches: list[tuple[float, float, int, list[str], str]] = []
    for embedding_score, bibcode in top_candidates:
        bibtex = database["joined_refs"][bibcode].get("bibtex", "")
        matched_tokens = [token for token in numeric_tokens if token in bibtex]
        numeric_score = sum(numeric_token_weight(token) for token in matched_tokens)
        normalized_numeric_score = numeric_score / total_numeric_weight
        reranked_matches.append(
            (embedding_score, normalized_numeric_score, numeric_score, matched_tokens, bibcode)
        )

    reranked_matches.sort(key=lambda item: (item[1], item[0]), reverse=True)
    return reranked_matches


def print_numeric_rerank_debug(reranked_matches: list[tuple[float, float, int, list[str], str]]) -> None:
    print("\nTop numeric rerank candidates")
    for rank, (embedding_score, normalized_numeric_score, numeric_score, matched_tokens, bibcode) in enumerate(
        reranked_matches[:TOP_K_FALLBACK], start=1
    ):
        print(f"{rank:2d}. {bibcode}")
        print(f"    embedding_score={embedding_score:.6f}")
        print(f"    numeric_score={numeric_score}")
        print(f"    normalized_numeric_score={normalized_numeric_score:.6f}")
        print(
            "    matched_numeric_tokens="
            + (", ".join(matched_tokens) if matched_tokens else "none")
        )


def print_match_result(
    bibcode: str,
    matched_entry: dict,
    path: str,
    extra_lines: list[str] | None = None,
) -> None:
    if DEBUG:
        print(f"Match path: {path}")
        if extra_lines:
            for line in extra_lines:
                print(line)
        print(f"Bibcode: {bibcode}")
        print(json.dumps(matched_entry, indent=2, ensure_ascii=False))
        return

    print(f"Bibcode: {bibcode}")
    print(json.dumps(matched_entry, indent=2, ensure_ascii=False))


def main() -> None:
    if not INPUT_TEXT:
        raise ValueError("Set INPUT_TEXT before running this script")

    database = load_database(DATABASE_PATH)
    arxiv_id = extract_arxiv_id(INPUT_TEXT)
    if arxiv_id is not None:
        arxiv_match = find_arxiv_match(database, arxiv_id)
        if arxiv_match is not None:
            bibcode, matched_entry = arxiv_match
            print_match_result(
                bibcode,
                matched_entry,
                path="arxiv",
                extra_lines=[f"Matched exact arXiv token: {arxiv_id}"],
            )
            return

    api_key = os.getenv("OPENROUTER_API_KEY_EMBEDDING")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY_EMBEDDING environment variable is not set")

    embedding_rows = load_embeddings(EMBEDDINGS_CSV)
    query_embedding = fetch_embeddings([INPUT_TEXT], api_key=api_key, model=MODEL)[0]

    scored_matches: list[tuple[float, str]] = []
    for bibcode, embedding in embedding_rows:
        score = cosine_similarity(query_embedding, embedding)
        scored_matches.append((score, bibcode))

    if not scored_matches:
        raise ValueError("Could not find a matching bibcode")

    scored_matches.sort(reverse=True)
    top_embedding_score, top_embedding_bibcode = scored_matches[0]
    top_embedding_falloff = embedding_falloff(scored_matches, rank=5)

    if DEBUG:
        print_top10_decay(scored_matches)
        print(
            f"\nEmbedding acceptance check: "
            f"top_score={top_embedding_score:.6f}, "
            f"falloff_top1_to_top5={top_embedding_falloff:.6f}"
        )

    if (
        top_embedding_score > EMBEDDING_ACCEPT_SCORE
        and top_embedding_falloff >= EMBEDDING_ACCEPT_FALLOFF
    ):
        print_match_result(
            top_embedding_bibcode,
            database["joined_refs"][top_embedding_bibcode],
            path="embedding",
            extra_lines=[
                f"embedding_score={top_embedding_score:.6f}",
                f"falloff_top1_to_top5={top_embedding_falloff:.6f}",
            ],
        )
        return

    numeric_tokens = extract_numeric_tokens(INPUT_TEXT)
    if DEBUG:
        if numeric_tokens:
            print("\nNumeric tokens from input: " + ", ".join(numeric_tokens))
        else:
            print("\nNumeric tokens from input: none")

    reranked_matches = rerank_top_matches_by_numeric_overlap(
        scored_matches, database=database, input_text=INPUT_TEXT
    )
    if DEBUG:
        print_numeric_rerank_debug(reranked_matches)

    top_numeric_match = reranked_matches[0]
    top_numeric_score = top_numeric_match[1]
    second_numeric_score = reranked_matches[1][1] if len(reranked_matches) > 1 else 0.0

    full_numeric_matches = [match for match in reranked_matches if match[1] == 1.0]
    if len(full_numeric_matches) == 1:
        winning_match = full_numeric_matches[0]
        print_match_result(
            winning_match[4],
            database["joined_refs"][winning_match[4]],
            path="numeric-all-match",
            extra_lines=[
                f"normalized_numeric_score={winning_match[1]:.6f}",
                f"embedding_score={winning_match[0]:.6f}",
                "matched_numeric_tokens=" + ", ".join(winning_match[3]),
            ],
        )
        return

    if top_numeric_score > NUMERIC_ACCEPT_SCORE and second_numeric_score < NUMERIC_REJECT_NEXT_SCORE:
        print_match_result(
            top_numeric_match[4],
            database["joined_refs"][top_numeric_match[4]],
            path="numeric-rerank",
            extra_lines=[
                f"normalized_numeric_score={top_numeric_match[1]:.6f}",
                f"next_normalized_numeric_score={second_numeric_score:.6f}",
                f"embedding_score={top_numeric_match[0]:.6f}",
                "matched_numeric_tokens="
                + (", ".join(top_numeric_match[3]) if top_numeric_match[3] else "none"),
            ],
        )
        return

    raise ValueError(
        "No confident match found. Embedding acceptance failed and numeric rerank remained ambiguous."
    )


if __name__ == "__main__":
    main()
