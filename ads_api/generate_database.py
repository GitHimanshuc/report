# %%
import requests
from urllib.parse import urlencode
import json
from pathlib import Path
import time
import os

API_KEY = os.getenv("ADS_API_KEY")
if API_KEY is None:
    raise ValueError("ADS_API_KEY environment variable is not set")


# %%


def get_bibtex_from_bibcode_list(bibcodes):
    payload = {"bibcode": bibcodes}
    serialized_payload = json.dumps(payload)
    results = requests.post(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        headers={"Authorization": "Bearer " + API_KEY},
        data=serialized_payload,
    )
    return results.json()


def get_bibtex_from_bibcode(bibcode):
    # Directly return the bibtex string for a single bibcode
    results = requests.get(
        f"https://api.adsabs.harvard.edu/v1/export/bibtex/{bibcode}",
        headers={"Authorization": "Bearer " + API_KEY},
    )
    return results.text


def update_database(database, database_path):
    # Call when new base paper is added

    bibcodes_set = set()
    for base_bibcode, data in database["base_papers"].items():
        bibcodes_set.update(data["bibcodes"])

    # Find the bibcodes that are not already in the joined_bibcodes set
    new_bibcodes = bibcodes_set - database["joined_bibcodes"]

    print(f"Found {len(new_bibcodes)} new bibcodes to add to the database")

    # get bibtex for the new bibcodes and add them to the joined_refs dictionary
    # create a dictionary with the payload values
    payload = {"bibcode": list(new_bibcodes)}

    # the json library offers an easy way to convert between JSON or dictionaries and their serialized strings
    serialized_payload = json.dumps(payload)

    results = requests.post(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        headers={"Authorization": "Bearer " + API_KEY},
        data=serialized_payload,
    )

    text = results.json()["export"]
    # split the text into individual bibtex entries and add them to the joined_refs dictionary
    for entry in text.split("\n\n"):
        if entry.strip() == "":
            continue
        bibcode = entry.split(",")[0].split("{")[1]
        database["joined_refs"][bibcode] = {"bibtex": entry}
    
    # Update the joined_bibcodes set with the new bibcodes
    database["joined_bibcodes"].update(new_bibcodes)


    # Unfortunately the bibcodes returned by the API for paper references is not in order so we can not use this

    # # Now each base paper entry has a key "indexed" which tells us whether it's bibcodes are matched with the entries in the joined_refs
    # # If not we go through each bibcode of that base paper and then add it's entry to all the relevant joined_refs aliases list
    # for base_bibcode, data in database["base_papers"].items():
    #     if "indexed" not in data:
    #         for i,bibcode in enumerate(data["bibcodes"]):
    #             database["joined_refs"][bibcode].setdefault("aliases", []).append(f"{data['identifier']}_{i}")

    #         data["indexed"] = True
        

    database_without_sets = database.copy()
    # Convert the set of bibcodes to a list for JSON serialization
    database_without_sets["joined_bibcodes"] = list(
        database_without_sets["joined_bibcodes"]
    )
    with open(database_path, "w") as f:
        json.dump(database_without_sets, f, indent=4, sort_keys=True)


# %%

# %%


database_path = Path("./database.json").resolve()

# have at least the latest sxs catalog
database = {
    "base_papers": {"2025CQGra..42s5017S": {"identifier": "__sxs25"}},
    "joined_bibcodes": [],
    "joined_refs": {},
}

if not database_path.exists():
    with open(database_path, "w") as f:
        json.dump(database, f, indent=4, sort_keys=True)
else:
    with open(database_path, "r") as f:
        database = json.load(f)
        # convert the bibcodes to a set for faster lookup
        database["joined_bibcodes"] = set(database["joined_bibcodes"])


# %%
# Consistency checks

# make sure each base paper has an identifier
for bibcode, info in database["base_papers"].items():
    assert "identifier" in info, f"Base paper {bibcode} is missing an identifier"

# make sure that the length of the bibcodes list matches the number of references in the refs dictionary
if len(database["joined_bibcodes"]) != len(database["joined_refs"]):
    raise ValueError(
        f"The number of bibcodes {len(database['joined_bibcodes'])} does not match the number of references {len(database['joined_refs'])} in the refs dictionary"
    )

# %%

# for each base paper, get the abstract and the bibcode of its references

# upto 2000 references per base paper is supported by the API
num_rows = 2000

database_updated = False  # Flag to track if the database was updated

for base_bibcode, info in database["base_papers"].items():
    # If the data is present in the database, skip the API call. Check for the bibcodes key
    if "bibcodes" in database["base_papers"][base_bibcode]:
        print(
            f"Data for {base_bibcode} is already present in the database, skipping API call"
        )
        continue

    else:
        print(
            f"Data for {base_bibcode} is not present in the database, making API call"
        )
        database_updated = True
        query = f"references(bibcode:{base_bibcode})"
        encoded_query = urlencode({"q": query, "fl": "bibcode,citation"})
        results = requests.get(
            f"https://api.adsabs.harvard.edu/v1/search/query?{encoded_query}&rows={num_rows}",
            headers={"Authorization": "Bearer " + API_KEY},
        )
        time.sleep(1)  # Be respectful to the API

        results_json = results.json()
        bibcodes = [doc["bibcode"] for doc in results_json["response"]["docs"]]

        database["base_papers"][base_bibcode]["bibcodes"] = bibcodes

        database["base_papers"][base_bibcode]["bibtex"] = get_bibtex_from_bibcode(
            base_bibcode
        )
        time.sleep(1)  # Be respectful to the API

if database_updated:
    update_database(database, database_path)

# %%
update_database(database, database_path)

# %%

