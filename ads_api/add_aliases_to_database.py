import argparse
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path

try:
    from embed_joined_refs import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MODEL,
        batched,
        bibtex_to_embedding_text,
        fetch_embeddings,
        load_existing_embeddings,
        sanitize_model_name,
        save_embeddings_csv,
    )
    from find_bibcode_match import load_embeddings, match_text
except ModuleNotFoundError:
    from ads_api.embed_joined_refs import (
        DEFAULT_BATCH_SIZE,
        DEFAULT_MODEL,
        batched,
        bibtex_to_embedding_text,
        fetch_embeddings,
        load_existing_embeddings,
        sanitize_model_name,
        save_embeddings_csv,
    )
    from ads_api.find_bibcode_match import load_embeddings, match_text


SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = SCRIPT_DIR / "database.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match per-paper numbered references and add aliases to database.json."
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=DATABASE_PATH,
        help="Path to the JSON database file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of bibtex entries to embed per API request when building per-paper CSV files.",
    )
    parser.add_argument(
        "--base-bibcode",
        action="append",
        default=[],
        help="Restrict processing to one or more base paper bibcodes.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed diagnostics for failed matches.",
    )
    parser.add_argument(
        "--regenerate-aliases",
        action="store_true",
        help="Reprocess papers even if base_papers['alises_generated'] is already true.",
    )
    return parser.parse_args()


def load_database(database_path: Path) -> dict:
    with database_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_database(database: dict, database_path: Path) -> None:
    database_to_save = dict(database)
    joined_bibcodes = database_to_save.get("joined_bibcodes", [])
    if isinstance(joined_bibcodes, set):
        database_to_save["joined_bibcodes"] = sorted(joined_bibcodes)
    with database_path.open("w", encoding="utf-8") as handle:
        json.dump(database_to_save, handle, indent=4, sort_keys=True)


def per_base_embedding_path(script_dir: Path, base_bibcode: str, model: str) -> Path:
    return script_dir / f"{base_bibcode}__{sanitize_model_name(model)}.csv"


def resolve_reference_txt_path(script_dir: Path, reference_txt_file: str) -> Path:
    reference_path = Path(reference_txt_file)
    if not reference_path.is_absolute():
        reference_path = (script_dir / reference_path).resolve()
    return reference_path


def collect_base_embedding_entries(database: dict, base_info: dict) -> list[tuple[str, str]]:
    joined_refs = database.get("joined_refs", {})
    entries: list[tuple[str, str]] = []
    missing_bibcodes: list[str] = []

    for bibcode in base_info.get("bibcodes", []):
        ref_entry = joined_refs.get(bibcode)
        if ref_entry is None:
            missing_bibcodes.append(bibcode)
            continue

        bibtex = ref_entry.get("bibtex")
        if not isinstance(bibtex, str) or not bibtex.strip():
            missing_bibcodes.append(bibcode)
            continue

        entries.append((bibcode, bibtex_to_embedding_text(bibtex)))

    if missing_bibcodes:
        preview = ", ".join(missing_bibcodes[:5])
        raise ValueError(f"Missing joined_refs bibtex for base paper references: {preview}")

    return entries


def ensure_base_paper_embedding_csv(
    database: dict,
    base_bibcode: str,
    base_info: dict,
    *,
    script_dir: Path,
    model: str,
    batch_size: int,
    api_key: str | None = None,
) -> Path:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_path = per_base_embedding_path(script_dir, base_bibcode, model)
    entries = collect_base_embedding_entries(database, base_info)
    entry_order = [bibcode for bibcode, _ in entries]
    current_bibcodes = set(entry_order)

    existing_rows, existing_format_ok = load_existing_embeddings(output_path)
    rows_to_keep: dict[str, tuple[str, list[float]]] = {}
    pending_entries: list[tuple[str, str]] = []

    for bibcode, embedding_text in entries:
        existing_row = existing_rows.get(bibcode)
        if existing_format_ok and existing_row is not None:
            rows_to_keep[bibcode] = existing_row
        else:
            pending_entries.append((bibcode, embedding_text))

    print(
        f"Embedding candidates for {base_bibcode}: "
        f"total={len(entries)}  existing={len(rows_to_keep)}  pending={len(pending_entries)}",
        flush=True,
    )

    needs_rewrite = (
        not output_path.exists()
        or not existing_format_ok
        or set(existing_rows) != current_bibcodes
        or bool(pending_entries)
    )

    if pending_entries:
        api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY_EMBEDDING")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY_EMBEDDING environment variable is not set")

        for batch in batched(pending_entries, batch_size):
            bibcodes = [bibcode for bibcode, _ in batch]
            texts = [text for _, text in batch]
            print(
                f"Embedding batch for {base_bibcode}: "
                f"size={len(batch)}  completed={len(rows_to_keep)}/{len(entries)}",
                flush=True,
            )
            embeddings = fetch_embeddings(texts, api_key=api_key, model=model)
            for bibcode, embedding in zip(bibcodes, embeddings, strict=True):
                rows_to_keep[bibcode] = (bibcode, embedding)

    if needs_rewrite:
        output_rows = [rows_to_keep[bibcode] for bibcode in entry_order if bibcode in rows_to_keep]
        save_embeddings_csv(output_rows, output_path)
        print(f"Wrote embeddings to {output_path}", flush=True)
    else:
        print(f"Embeddings already up to date at {output_path}", flush=True)

    return output_path


def ensure_base_embeddings_for_database(
    database: dict,
    *,
    script_dir: Path = SCRIPT_DIR,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    api_key: str | None = None,
    base_bibcodes: set[str] | None = None,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    for base_bibcode, base_info in database.get("base_papers", {}).items():
        if base_bibcodes is not None and base_bibcode not in base_bibcodes:
            continue
        if "reference_txt_file" not in base_info:
            continue
        if "bibcodes" not in base_info:
            raise ValueError(f"Base paper {base_bibcode} is missing its bibcodes list")

        output_paths[base_bibcode] = ensure_base_paper_embedding_csv(
            database,
            base_bibcode,
            base_info,
            script_dir=script_dir,
            model=model,
            batch_size=batch_size,
            api_key=api_key,
        )

    return output_paths


def subset_database_for_bibcodes(database: dict, bibcodes: list[str]) -> dict:
    return {
        "joined_refs": {
            bibcode: database["joined_refs"][bibcode]
            for bibcode in bibcodes
            if bibcode in database["joined_refs"]
        }
    }


def remove_alias_from_joined_refs(database: dict, alias: str) -> None:
    for entry in database.get("joined_refs", {}).values():
        aliases = entry.get("aliases")
        if not isinstance(aliases, list) or alias not in aliases:
            continue
        entry["aliases"] = [existing_alias for existing_alias in aliases if existing_alias != alias]
        if not entry["aliases"]:
            del entry["aliases"]


def find_alias_owner(database: dict, alias: str) -> str | None:
    for existing_bibcode, entry in database.get("joined_refs", {}).items():
        aliases = entry.get("aliases")
        if isinstance(aliases, list) and alias in aliases:
            return existing_bibcode
    return None


def add_alias_to_bibcode(database: dict, bibcode: str, alias: str) -> str:
    existing_owner = find_alias_owner(database, alias)
    if existing_owner == bibcode:
        return "unchanged"

    remove_alias_from_joined_refs(database, alias)
    aliases = database["joined_refs"][bibcode].setdefault("aliases", [])
    if alias not in aliases:
        aliases.append(alias)
    if existing_owner is None:
        return "added"
    return "moved"


def iter_reference_lines(reference_txt_path: Path) -> list[tuple[int, str]]:
    lines = [
        line.strip()
        for line in reference_txt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return list(enumerate(lines, start=1))


def should_process_base_paper(base_info: dict, *, regenerate_aliases: bool) -> bool:
    if "reference_txt_file" not in base_info:
        return False
    if regenerate_aliases:
        return True
    return not bool(base_info.get("alises_generated"))


def process_base_paper_aliases(
    database: dict,
    base_bibcode: str,
    base_info: dict,
    *,
    script_dir: Path,
    model: str,
    api_key: str,
    debug: bool,
) -> tuple[int, int, list[str]]:
    identifier = base_info.get("identifier")
    if not identifier:
        raise ValueError(f"Base paper {base_bibcode} is missing an identifier")

    reference_txt_file = base_info.get("reference_txt_file")
    if not reference_txt_file:
        return 0, 0, []

    reference_txt_path = resolve_reference_txt_path(script_dir, reference_txt_file)
    if not reference_txt_path.exists():
        raise FileNotFoundError(f"Reference text file not found: {reference_txt_path}")

    embedding_path = per_base_embedding_path(script_dir, base_bibcode, model)
    embedding_rows = load_embeddings(embedding_path)
    restricted_database = subset_database_for_bibcodes(database, base_info["bibcodes"])
    reference_lines = iter_reference_lines(reference_txt_path)

    success_count = 0
    failure_count = 0
    alias_added_count = 0
    alias_moved_count = 0
    alias_unchanged_count = 0
    failure_messages: list[str] = []

    print(
        f"Processing {len(reference_lines)} references for {base_bibcode} from {reference_txt_path}",
        flush=True,
    )

    for entry_number, input_text in reference_lines:
        alias = f"{identifier}_{entry_number}"
        debug_buffer = io.StringIO()
        try:
            with redirect_stdout(debug_buffer):
                matched_bibcode, _, match_path, _ = match_text(
                    input_text,
                    debug=debug,
                    database=restricted_database,
                    embedding_rows=embedding_rows,
                    api_key=api_key,
                )
            alias_status = add_alias_to_bibcode(database, matched_bibcode, alias)
            if alias_status == "added":
                alias_added_count += 1
            elif alias_status == "moved":
                alias_moved_count += 1
            else:
                alias_unchanged_count += 1
            success_count += 1
            print(
                f"[{entry_number}/{len(reference_lines)}] OK  "
                f"success={success_count}  failed={failure_count}  "
                f"alias={alias}  bibcode={matched_bibcode}  path={match_path}",
                flush=True,
            )
        except Exception as error:
            failure_count += 1
            message_lines = [f"{alias}: {input_text}", f"  {type(error).__name__}: {error}"]
            if debug:
                debug_output = debug_buffer.getvalue().strip()
                if debug_output:
                    for line in debug_output.splitlines():
                        message_lines.append(f"  {line}")
            failure_messages.append("\n".join(message_lines))
            print(
                f"[{entry_number}/{len(reference_lines)}] ERR  "
                f"success={success_count}  failed={failure_count}  alias={alias}",
                flush=True,
            )

    print(
        f"Alias summary for {base_bibcode}: "
        f"added={alias_added_count}  moved={alias_moved_count}  "
        f"unchanged={alias_unchanged_count}  failed={failure_count}",
        flush=True,
    )

    return success_count, failure_count, failure_messages


def main() -> None:
    args = parse_args()
    database_path = args.database.resolve()
    database = load_database(database_path)

    requested_base_bibcodes = set(args.base_bibcode) if args.base_bibcode else None
    if requested_base_bibcodes is not None:
        missing_base_bibcodes = requested_base_bibcodes - set(database.get("base_papers", {}))
        if missing_base_bibcodes:
            missing_list = ", ".join(sorted(missing_base_bibcodes))
            raise ValueError(f"Unknown base paper bibcode(s): {missing_list}")

    base_bibcodes_to_process: set[str] = set()
    skipped_base_bibcodes: list[str] = []
    for base_bibcode, base_info in database.get("base_papers", {}).items():
        if requested_base_bibcodes is not None and base_bibcode not in requested_base_bibcodes:
            continue
        if not should_process_base_paper(
            base_info,
            regenerate_aliases=args.regenerate_aliases,
        ):
            if "reference_txt_file" in base_info:
                skipped_base_bibcodes.append(base_bibcode)
            continue
        if "reference_txt_file" in base_info:
            base_bibcodes_to_process.add(base_bibcode)

    api_key = os.getenv("OPENROUTER_API_KEY_EMBEDDING")
    if base_bibcodes_to_process:
        ensure_base_embeddings_for_database(
            database,
            script_dir=database_path.parent,
            model=DEFAULT_MODEL,
            batch_size=args.batch_size,
            api_key=api_key,
            base_bibcodes=base_bibcodes_to_process,
        )
    else:
        print("No base papers need alias generation.", flush=True)

    if skipped_base_bibcodes:
        print(
            "Skipping base papers with alises_generated=true: "
            + ", ".join(sorted(skipped_base_bibcodes)),
            flush=True,
        )

    total_success = 0
    total_failure = 0
    all_failures: list[str] = []

    for base_bibcode, base_info in database.get("base_papers", {}).items():
        if base_bibcode not in base_bibcodes_to_process:
            continue

        success_count, failure_count, failure_messages = process_base_paper_aliases(
            database,
            base_bibcode,
            base_info,
            script_dir=database_path.parent,
            model=DEFAULT_MODEL,
            api_key=api_key,
            debug=args.debug,
        )
        base_info["alises_generated"] = failure_count == 0
        total_success += success_count
        total_failure += failure_count
        all_failures.extend([f"{base_bibcode}\n{message}" for message in failure_messages])

    save_database(database, database_path)

    print(f"Completed alias update: success={total_success}, failed={total_failure}", flush=True)
    if all_failures:
        print("\nFailures:")
        print("\n\n".join(all_failures))


if __name__ == "__main__":
    main()
