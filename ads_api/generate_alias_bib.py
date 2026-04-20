import argparse
import json
import re
from pathlib import Path


ENTRY_HEADER_RE = re.compile(r"^(@[A-Za-z]+)\{([^,]+),", re.MULTILINE)


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


def iter_alias_entries(database: dict) -> list[str]:
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
            output_key = f"{bibcode}_{alias}"
            output_entries.append(replace_bibtex_key(bibtex, output_key))

    return output_entries


def main() -> None:
    args = parse_args()
    database = load_database(args.database)
    output_entries = iter_alias_entries(database)

    args.output.write_text("\n\n".join(output_entries) + ("\n" if output_entries else ""), encoding="utf-8")

    print(f"Wrote {len(output_entries)} alias BibTeX entries to {args.output}")


if __name__ == "__main__":
    main()
