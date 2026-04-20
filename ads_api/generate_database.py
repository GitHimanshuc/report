import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode

import requests

try:
    from add_aliases_to_database import ensure_base_embeddings_for_database
except ModuleNotFoundError:
    from ads_api.add_aliases_to_database import ensure_base_embeddings_for_database


SCRIPT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = SCRIPT_DIR / "database.json"


def get_api_key() -> str:
    api_key = os.getenv("ADS_API_KEY")
    if api_key is None:
        raise ValueError("ADS_API_KEY environment variable is not set")
    return api_key


def get_bibtex_from_bibcode_list(bibcodes: list[str], *, api_key: str) -> dict:
    payload = {"bibcode": bibcodes}
    serialized_payload = json.dumps(payload)
    results = requests.post(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        headers={"Authorization": "Bearer " + api_key},
        data=serialized_payload,
    )
    return results.json()


def get_bibtex_from_bibcode(bibcode: str, *, api_key: str) -> str:
    results = requests.get(
        f"https://api.adsabs.harvard.edu/v1/export/bibtex/{bibcode}",
        headers={"Authorization": "Bearer " + api_key},
    )
    return results.text


def save_database(database: dict, database_path: Path) -> None:
    database_to_save = dict(database)
    joined_bibcodes = database_to_save.get("joined_bibcodes", [])
    if isinstance(joined_bibcodes, set):
        database_to_save["joined_bibcodes"] = sorted(joined_bibcodes)
    with database_path.open("w", encoding="utf-8") as handle:
        json.dump(database_to_save, handle, indent=4, sort_keys=True)


def load_database(database_path: Path) -> dict:
    default_database = {
        "base_papers": {"2025CQGra..42s5017S": {"identifier": "__sxs25"}},
        "joined_bibcodes": [],
        "joined_refs": {},
    }

    if not database_path.exists():
        save_database(default_database, database_path)
        database = default_database
    else:
        with database_path.open("r", encoding="utf-8") as handle:
            database = json.load(handle)

    database["joined_bibcodes"] = set(database.get("joined_bibcodes", []))
    return database


def update_database(database: dict, database_path: Path, *, api_key: str) -> None:
    bibcodes_set: set[str] = set()
    for data in database["base_papers"].values():
        bibcodes_set.update(data["bibcodes"])

    new_bibcodes = bibcodes_set - database["joined_bibcodes"]
    print(f"Found {len(new_bibcodes)} new bibcodes to add to the database")

    if not new_bibcodes:
        save_database(database, database_path)
        return

    payload = {"bibcode": list(new_bibcodes)}
    serialized_payload = json.dumps(payload)

    results = requests.post(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        headers={"Authorization": "Bearer " + api_key},
        data=serialized_payload,
    )

    text = results.json()["export"]
    for entry in text.split("\n\n"):
        if not entry.strip():
            continue
        bibcode = entry.split(",")[0].split("{")[1]
        database["joined_refs"][bibcode] = {"bibtex": entry}

    database["joined_bibcodes"].update(new_bibcodes)
    save_database(database, database_path)


def run_consistency_checks(database: dict) -> None:
    for bibcode, info in database["base_papers"].items():
        if "identifier" not in info:
            raise ValueError(f"Base paper {bibcode} is missing an identifier")

    if len(database["joined_bibcodes"]) != len(database["joined_refs"]):
        raise ValueError(
            f"The number of bibcodes {len(database['joined_bibcodes'])} does not match "
            f"the number of references {len(database['joined_refs'])} in the refs dictionary"
        )


def populate_missing_base_paper_data(
    database: dict,
    *,
    api_key: str,
    num_rows: int = 2000,
) -> bool:
    database_updated = False

    for base_bibcode, info in database["base_papers"].items():
        if "bibcodes" in info:
            print(f"Data for {base_bibcode} is already present in the database, skipping API call")
            continue

        print(f"Data for {base_bibcode} is not present in the database, making API call")
        database_updated = True
        query = f"references(bibcode:{base_bibcode})"
        encoded_query = urlencode({"q": query, "fl": "bibcode,citation"})
        results = requests.get(
            f"https://api.adsabs.harvard.edu/v1/search/query?{encoded_query}&rows={num_rows}",
            headers={"Authorization": "Bearer " + api_key},
        )
        time.sleep(1)

        results_json = results.json()
        info["bibcodes"] = [doc["bibcode"] for doc in results_json["response"]["docs"]]
        info["bibtex"] = get_bibtex_from_bibcode(base_bibcode, api_key=api_key)
        time.sleep(1)

    return database_updated


def main() -> None:
    api_key = get_api_key()
    database = load_database(DATABASE_PATH)
    run_consistency_checks(database)
    populate_missing_base_paper_data(database, api_key=api_key)
    update_database(database, DATABASE_PATH, api_key=api_key)
    ensure_base_embeddings_for_database(database, script_dir=SCRIPT_DIR)


if __name__ == "__main__":
    main()
