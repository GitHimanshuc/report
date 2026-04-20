import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Iterable

import requests


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "openai/text-embedding-3-small"
DEFAULT_BATCH_SIZE = 128
IGNORED_BIBTEX_FIELDS = {"adsnote", "adsurl", "month", "url"}


def sanitize_model_name(model: str) -> str:
    return model.replace("/", "__")


def default_output_path(script_dir: Path, model: str) -> Path:
    return script_dir / f"{sanitize_model_name(model)}.csv"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Create embeddings for each entry in database.json['joined_refs']."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=script_dir / "database.json",
        help="Path to the JSON database file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output CSV file. Defaults to a filename derived from the model.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_MODEL),
        help="Embedding model to request from OpenRouter.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of joined_refs entries to embed per API request.",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = default_output_path(script_dir, args.model)
    return args


def load_database(database_path: Path) -> dict:
    with database_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_top_level_comma_separated(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    brace_depth = 0
    in_quotes = False

    for character in text:
        if character == '"' and (not current or current[-1] != "\\"):
            in_quotes = not in_quotes
        elif not in_quotes:
            if character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth = max(brace_depth - 1, 0)
            elif character == "," and brace_depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue

        current.append(character)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def unwrap_bibtex_value(value: str) -> str:
    text = value.strip()
    while len(text) >= 2:
        if text[0] == '"' and text[-1] == '"':
            text = text[1:-1].strip()
            continue
        if text[0] == "{" and text[-1] == "}":
            text = text[1:-1].strip()
            continue
        break

    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_n_authors(author_value: str, count: int = 5) -> str:
    authors = re.split(r"\s+and\s+", author_value)
    cleaned_authors = [author.strip() for author in authors if author.strip()]
    if not cleaned_authors:
        return author_value.strip()
    return " and ".join(cleaned_authors[:count])


def bibtex_to_embedding_text(bibtex_entry: str) -> str:
    first_comma = bibtex_entry.find(",")
    last_closing_brace = bibtex_entry.rfind("}")
    if first_comma == -1 or last_closing_brace == -1 or first_comma >= last_closing_brace:
        return re.sub(r"\s+", " ", bibtex_entry).strip()

    fields_blob = bibtex_entry[first_comma + 1 : last_closing_brace]
    field_chunks = split_top_level_comma_separated(fields_blob)

    values: list[str] = []
    for chunk in field_chunks:
        if "=" not in chunk:
            continue

        key, raw_value = chunk.split("=", maxsplit=1)
        cleaned_key = key.strip().lower()
        if cleaned_key in IGNORED_BIBTEX_FIELDS:
            continue
        cleaned_value = unwrap_bibtex_value(raw_value)
        if not cleaned_value:
            continue

        if cleaned_key == "author":
            cleaned_value = first_n_authors(cleaned_value)

        values.append(cleaned_value)

    return " ".join(values)


def collect_bibtex_entries(joined_refs: dict) -> list[tuple[str, str]]:
    bibtex_entries: list[tuple[str, str]] = []
    for bibcode, ref_entry in sorted(joined_refs.items()):
        bibtex = ref_entry.get("bibtex")
        if isinstance(bibtex, str) and bibtex.strip():
            bibtex_entries.append((bibcode, bibtex_to_embedding_text(bibtex)))
    return bibtex_entries


def batched(items: list[tuple[str, str]], batch_size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def fetch_embeddings(texts: list[str], api_key: str, model: str) -> list[list[float]]:
    response = requests.post(
        OPENROUTER_EMBEDDINGS_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "input": texts,
            "encoding_format": "float",
        },
        timeout=120,
    )
    response.raise_for_status()

    payload = response.json()
    data = sorted(payload["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


def load_existing_embeddings(
    output_path: Path,
) -> tuple[dict[str, tuple[str, list[float]]], bool]:
    if not output_path.exists():
        return {}, True

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        required_columns = {"bibcode"}
        if not required_columns.issubset(fieldnames):
            return {}, False

        dim_columns = [name for name in fieldnames if name.startswith("dim_")]
        dim_columns.sort(key=lambda name: int(name.split("_", maxsplit=1)[1]))

        rows: dict[str, tuple[str, list[float]]] = {}
        for row in reader:
            rows[row["bibcode"]] = (
                row["bibcode"],
                [float(row[column]) for column in dim_columns],
            )

    return rows, True


def save_embeddings_csv(
    rows: list[tuple[str, list[float]]],
    output_path: Path,
) -> None:
    if not rows:
        raise ValueError("No embeddings were generated")

    embedding_dim = len(rows[0][1])
    header = ["bibcode", *[f"dim_{i}" for i in range(embedding_dim)]]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for bibcode, embedding in rows:
            writer.writerow([bibcode, *embedding])


def main() -> None:
    args = parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY_EMBEDDING")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY_EMBEDDING environment variable is not set")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    database = load_database(args.database)
    joined_refs = database.get("joined_refs", {})
    bibtex_entries = collect_bibtex_entries(joined_refs)

    existing_rows, existing_format_ok = load_existing_embeddings(args.output)
    if args.output.exists() and not existing_format_ok:
        print(
            "Existing embeddings file does not match the current format; "
            "it will be regenerated."
        )

    rows_to_keep: dict[str, tuple[str, list[float]]] = {}
    pending_entries: list[tuple[str, str]] = []
    for bibcode, bibtex in bibtex_entries:
        existing_row = existing_rows.get(bibcode)
        if existing_row is not None:
            rows_to_keep[bibcode] = existing_row
        else:
            pending_entries.append((bibcode, bibtex))

    output_rows: list[tuple[str, list[float]]] = list(rows_to_keep.values())
    for batch in batched(pending_entries, args.batch_size):
        bibcodes = [bibcode for bibcode, _ in batch]
        texts = [text for _, text in batch]
        embeddings = fetch_embeddings(texts, api_key=api_key, model=args.model)

        for bibcode, _, embedding in zip(bibcodes, texts, embeddings, strict=True):
            output_rows.append((bibcode, embedding))

        print(
            f"Embedded {len(output_rows)}/{len(bibtex_entries)} bibtex entries "
            f"({len(pending_entries)} needed updates)"
        )

    output_rows.sort(key=lambda row: row[0])
    save_embeddings_csv(output_rows, args.output)
    print(f"Wrote {len(output_rows)} embeddings to {args.output}")


if __name__ == "__main__":
    main()
