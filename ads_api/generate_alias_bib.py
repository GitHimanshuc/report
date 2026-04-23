import argparse
import json
import re
from pathlib import Path


ENTRY_HEADER_RE = re.compile(r"^(@[A-Za-z]+)\{([^,]+),", re.MULTILINE)
AUTHOR_TRUNCATION_THRESHOLD = 5
AUTHOR_TRUNCATION_KEEP = 3


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a .bib file with one repeated entry per alias in database.json."
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
        default=script_dir / "aliases.bib",
        help="Path to the output .bib file.",
    )
    parser.add_argument(
        "--truncate-long-author-lists",
        action="store_true",
        help=(
            "If set, author fields with more than "
            f"{AUTHOR_TRUNCATION_THRESHOLD} authors are truncated to the first "
            f"{AUTHOR_TRUNCATION_KEEP} authors followed by 'others'."
        ),
    )
    return parser.parse_args()


def load_database(database_path: Path) -> dict:
    with database_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def replace_bibtex_key(bibtex_entry: str, new_key: str) -> str:
    match = ENTRY_HEADER_RE.search(bibtex_entry)
    if match is None:
        raise ValueError(f"Could not parse BibTeX entry header: {bibtex_entry[:80]!r}")

    entry_type = match.group(1)
    original_key = match.group(2)
    return bibtex_entry.replace(f"{entry_type}{{{original_key},", f"{entry_type}{{{new_key},", 1)


def find_field_value_span(bibtex_entry: str, field_name: str) -> tuple[int, int] | None:
    field_match = re.search(rf"^[ \t]*{re.escape(field_name)}[ \t]*=[ \t]*", bibtex_entry, re.MULTILINE)
    if field_match is None:
        return None

    value_start = field_match.end()
    while value_start < len(bibtex_entry) and bibtex_entry[value_start].isspace():
        value_start += 1

    if value_start >= len(bibtex_entry):
        return None

    delimiter = bibtex_entry[value_start]
    if delimiter == "{":
        depth = 0
        cursor = value_start
        while cursor < len(bibtex_entry):
            current_char = bibtex_entry[cursor]
            if current_char == "{":
                depth += 1
            elif current_char == "}":
                depth -= 1
                if depth == 0:
                    return value_start, cursor + 1
            cursor += 1
        raise ValueError(f"Unbalanced braces in {field_name!r} field: {bibtex_entry[:80]!r}")

    if delimiter == '"':
        cursor = value_start + 1
        escaped = False
        while cursor < len(bibtex_entry):
            current_char = bibtex_entry[cursor]
            if escaped:
                escaped = False
            elif current_char == "\\":
                escaped = True
            elif current_char == '"':
                return value_start, cursor + 1
            cursor += 1
        raise ValueError(f'Unbalanced quotes in {field_name!r} field: {bibtex_entry[:80]!r}')

    cursor = value_start
    while cursor < len(bibtex_entry) and bibtex_entry[cursor] not in ",\n\r":
        cursor += 1
    return value_start, cursor


def unwrap_bibtex_value(raw_value: str) -> str:
    if len(raw_value) >= 2 and raw_value[0] == "{" and raw_value[-1] == "}":
        return raw_value[1:-1]
    if len(raw_value) >= 2 and raw_value[0] == '"' and raw_value[-1] == '"':
        return raw_value[1:-1]
    return raw_value


def wrap_bibtex_value(original_raw_value: str, inner_value: str) -> str:
    if len(original_raw_value) >= 2 and original_raw_value[0] == "{" and original_raw_value[-1] == "}":
        return "{" + inner_value + "}"
    if len(original_raw_value) >= 2 and original_raw_value[0] == '"' and original_raw_value[-1] == '"':
        return '"' + inner_value + '"'
    return inner_value


def split_bibtex_names(author_value: str) -> list[str]:
    names: list[str] = []
    current_name: list[str] = []
    brace_depth = 0
    cursor = 0

    while cursor < len(author_value):
        if brace_depth == 0 and author_value.startswith(" and ", cursor):
            names.append("".join(current_name).strip())
            current_name = []
            cursor += len(" and ")
            continue

        current_char = author_value[cursor]
        if current_char == "{":
            brace_depth += 1
        elif current_char == "}":
            brace_depth -= 1
        current_name.append(current_char)
        cursor += 1

    trailing_name = "".join(current_name).strip()
    if trailing_name:
        names.append(trailing_name)

    return names


def truncate_author_list(bibtex_entry: str) -> str:
    author_span = find_field_value_span(bibtex_entry, "author")
    if author_span is None:
        return bibtex_entry

    value_start, value_end = author_span
    raw_author_value = bibtex_entry[value_start:value_end]
    author_names = split_bibtex_names(unwrap_bibtex_value(raw_author_value).strip())
    if len(author_names) <= AUTHOR_TRUNCATION_THRESHOLD:
        return bibtex_entry

    truncated_authors = " and ".join(author_names[:AUTHOR_TRUNCATION_KEEP] + ["others"])
    updated_raw_value = wrap_bibtex_value(raw_author_value, truncated_authors)
    return bibtex_entry[:value_start] + updated_raw_value + bibtex_entry[value_end:]


def prepare_bibtex_entry(
    bibtex_entry: str,
    new_key: str,
    *,
    truncate_long_author_lists: bool,
) -> str:
    updated_entry = replace_bibtex_key(bibtex_entry, new_key)
    if truncate_long_author_lists:
        updated_entry = truncate_author_list(updated_entry)
    return updated_entry


def iter_alias_entries(database: dict, *, truncate_long_author_lists: bool) -> list[str]:
    output_entries: list[str] = []

    for bibcode, ref_entry in sorted(database.get("joined_refs", {}).items()):
        bibtex = ref_entry.get("bibtex")
        aliases = ref_entry.get("aliases")

        if not isinstance(bibtex, str) or not bibtex.strip():
            continue
        if not isinstance(aliases, list):
            continue

        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                continue
            output_key = f"{alias[2:]}"
            output_entries.append(
                prepare_bibtex_entry(
                    bibtex,
                    output_key,
                    truncate_long_author_lists=truncate_long_author_lists,
                )
            )

    return output_entries


def main() -> None:
    args = parse_args()
    database = load_database(args.database)
    output_entries = iter_alias_entries(
        database,
        truncate_long_author_lists=args.truncate_long_author_lists,
    )

    args.output.write_text("\n\n".join(output_entries) + ("\n" if output_entries else ""), encoding="utf-8")

    print(f"Wrote {len(output_entries)} alias BibTeX entries to {args.output}")


if __name__ == "__main__":
    main()
