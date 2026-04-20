import argparse
import csv
import io
import json
import math
import os
import re
from contextlib import redirect_stdout
from pathlib import Path

try:
    from embed_joined_refs import default_output_path, fetch_embeddings
except ModuleNotFoundError:
    from ads_api.embed_joined_refs import default_output_path, fetch_embeddings

MODEL = "openai/text-embedding-3-small"
TOP_K_FALLBACK = 5
EMBEDDING_ACCEPT_SCORE = 0.75
EMBEDDING_ACCEPT_FALLOFF = 0.20
NUMERIC_ACCEPT_SCORE = 0.8
NUMERIC_REJECT_NEXT_SCORE = 0.5
MIN_EMBEDDING_SCORE_FOR_NUMERIC_RERANK = 0.60
MIN_NUMERIC_ALL_MATCH_SCORE = 5
MIN_NUMERIC_TOKEN_VALUE = 50
SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = SCRIPT_DIR / "database.json"
EMBEDDINGS_CSV = default_output_path(SCRIPT_DIR, MODEL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match each reference in a text file against ads_api/database.json."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a .txt file containing one reference per line.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include detailed matching diagnostics in the output report.",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Write only failed matches to the output report.",
    )
    return parser.parse_args()


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
    match = re.search(r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?", text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1)


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
        try:
            numeric_value = float(token)
        except ValueError:
            continue
        if numeric_value < MIN_NUMERIC_TOKEN_VALUE:
            continue
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
    *,
    excluded_tokens: set[str] | None = None,
) -> list[tuple[float, float, int, list[str], str]]:
    numeric_tokens = extract_numeric_tokens(input_text)
    if excluded_tokens:
        numeric_tokens = [token for token in numeric_tokens if token not in excluded_tokens]
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


def match_text(
    input_text: str,
    *,
    debug: bool | None = None,
    database: dict | None = None,
    embedding_rows: list[tuple[str, list[float]]] | None = None,
    api_key: str | None = None,
) -> tuple[str, dict, str, list[str]]:
    if not input_text:
        raise ValueError("Input text is empty")

    debug_enabled = bool(debug)
    database = database if database is not None else load_database(DATABASE_PATH)
    arxiv_id = extract_arxiv_id(input_text)
    excluded_numeric_tokens: set[str] = set()
    if arxiv_id is not None:
        arxiv_match = find_arxiv_match(database, arxiv_id)
        if arxiv_match is not None:
            bibcode, matched_entry = arxiv_match
            return (
                bibcode,
                matched_entry,
                "arxiv",
                [f"Matched exact arXiv token: {arxiv_id}"],
            )
        excluded_numeric_tokens.add(arxiv_id)

    api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY_EMBEDDING")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY_EMBEDDING environment variable is not set")

    embedding_rows = embedding_rows if embedding_rows is not None else load_embeddings(EMBEDDINGS_CSV)
    query_embedding = fetch_embeddings([input_text], api_key=api_key, model=MODEL)[0]

    scored_matches: list[tuple[float, str]] = []
    for bibcode, embedding in embedding_rows:
        score = cosine_similarity(query_embedding, embedding)
        scored_matches.append((score, bibcode))

    if not scored_matches:
        raise ValueError("Could not find a matching bibcode")

    scored_matches.sort(reverse=True)
    top_embedding_score, top_embedding_bibcode = scored_matches[0]
    top_embedding_falloff = embedding_falloff(scored_matches, rank=5)

    if debug_enabled:
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
        return (
            top_embedding_bibcode,
            database["joined_refs"][top_embedding_bibcode],
            "embedding",
            [
                f"embedding_score={top_embedding_score:.6f}",
                f"falloff_top1_to_top5={top_embedding_falloff:.6f}",
            ],
        )

    if top_embedding_score < MIN_EMBEDDING_SCORE_FOR_NUMERIC_RERANK:
        raise ValueError(
            "No confident match found. Top embedding score was below the numeric-rerank threshold."
        )

    numeric_tokens = extract_numeric_tokens(input_text)
    if debug_enabled:
        if numeric_tokens:
            print("\nNumeric tokens from input: " + ", ".join(numeric_tokens))
        else:
            print("\nNumeric tokens from input: none")

    reranked_matches = rerank_top_matches_by_numeric_overlap(
        scored_matches,
        database=database,
        input_text=input_text,
        excluded_tokens=excluded_numeric_tokens,
    )
    if debug_enabled:
        print_numeric_rerank_debug(reranked_matches)

    top_numeric_match = reranked_matches[0]
    top_numeric_score = top_numeric_match[1]
    second_numeric_score = reranked_matches[1][1] if len(reranked_matches) > 1 else 0.0

    full_numeric_matches = [
        match
        for match in reranked_matches
        if match[1] == 1.0 and match[2] >= MIN_NUMERIC_ALL_MATCH_SCORE
    ]
    if len(full_numeric_matches) == 1:
        winning_match = full_numeric_matches[0]
        return (
            winning_match[4],
            database["joined_refs"][winning_match[4]],
            "numeric-all-match",
            [
                f"normalized_numeric_score={winning_match[1]:.6f}",
                f"embedding_score={winning_match[0]:.6f}",
                "matched_numeric_tokens=" + ", ".join(winning_match[3]),
            ],
        )

    if top_numeric_score > NUMERIC_ACCEPT_SCORE and second_numeric_score < NUMERIC_REJECT_NEXT_SCORE:
        return (
            top_numeric_match[4],
            database["joined_refs"][top_numeric_match[4]],
            "numeric-rerank",
            [
                f"normalized_numeric_score={top_numeric_match[1]:.6f}",
                f"next_normalized_numeric_score={second_numeric_score:.6f}",
                f"embedding_score={top_numeric_match[0]:.6f}",
                "matched_numeric_tokens="
                + (", ".join(top_numeric_match[3]) if top_numeric_match[3] else "none"),
            ],
        )

    raise ValueError(
        "No confident match found. Embedding acceptance failed and numeric rerank remained ambiguous."
    )


def format_match_block(
    input_text: str,
    bibcode: str,
    matched_entry: dict,
    path: str,
    extra_lines: list[str],
    debug_output: str,
    *,
    debug: bool,
) -> str:
    lines = [f"Query: {input_text}"]
    if debug:
        lines.append(f"Match path: {path}")
        lines.extend(extra_lines)
        debug_output = debug_output.strip()
        if debug_output:
            lines.append(debug_output)
    lines.append(f"Bibcode: {bibcode}")
    lines.append(json.dumps(matched_entry, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def format_error_block(input_text: str, error: Exception, debug_output: str, *, debug: bool) -> str:
    lines = [f"Query: {input_text}", f"Error: {type(error).__name__}: {error}"]
    if debug:
        debug_output = debug_output.strip()
        if debug_output:
            lines.append(debug_output)
    return "\n".join(lines)


def output_path_for_input(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_OUTPUT.txt")


def print_progress(index: int, total: int, success_count: int, failure_count: int, status: str) -> None:
    print(
        f"[{index}/{total}] {status}  "
        f"success={success_count}  "
        f"failed={failure_count}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    input_path = args.input_file.resolve()
    if input_path.suffix.lower() != ".txt":
        raise ValueError("Input file must have a .txt extension")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    database = load_database(DATABASE_PATH)
    embedding_rows = load_embeddings(EMBEDDINGS_CSV)
    api_key = os.getenv("OPENROUTER_API_KEY_EMBEDDING")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY_EMBEDDING environment variable is not set")

    input_lines = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not input_lines:
        raise ValueError(f"No non-empty lines found in {input_path}")

    output_blocks: list[str] = []
    separator = "\n\n" + ("=" * 100) + "\n\n"
    total_count = len(input_lines)
    success_count = 0
    failure_count = 0

    print(f"Processing {total_count} references from {input_path}", flush=True)

    for index, input_text in enumerate(input_lines, start=1):
        debug_buffer = io.StringIO()
        try:
            with redirect_stdout(debug_buffer):
                bibcode, matched_entry, path, extra_lines = match_text(
                    input_text,
                    debug=args.debug,
                    database=database,
                    embedding_rows=embedding_rows,
                    api_key=api_key,
                )
            if not args.only_failed:
                output_blocks.append(
                    format_match_block(
                        input_text,
                        bibcode,
                        matched_entry,
                        path,
                        extra_lines,
                        debug_buffer.getvalue(),
                        debug=args.debug,
                    )
                )
            success_count += 1
            print_progress(index, total_count, success_count, failure_count, "OK")
        except Exception as error:
            output_blocks.append(
                format_error_block(
                    input_text,
                    error,
                    debug_buffer.getvalue(),
                    debug=args.debug,
                )
            )
            failure_count += 1
            print_progress(index, total_count, success_count, failure_count, "ERR")

    output_path = output_path_for_input(input_path)
    output_text = separator.join(output_blocks)
    if output_text:
        output_text += "\n"
    output_path.write_text(output_text, encoding="utf-8")
    print(
        f"Completed {total_count} references: "
        f"success={success_count}, failed={failure_count}",
        flush=True,
    )
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
