import os
import json


def append_json(file: str, data: dict):
    """
    Appends data to a JSON file.

    If the file does not exist, it is created.
    New data is merged with existing data, with new keys overwriting old ones.
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)

    if os.path.exists(file):
        with open(file, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}

    existing_data.update(data)

    with open(file, "w") as f:
        json.dump(existing_data, f, indent=4)


def append_meta(path: str, data: dict):
    """Append metadata to a meta.json file in the given path."""
    meta_file = os.path.join(path, "meta.json")
    append_json(meta_file, data)


def load_meta(path: str) -> dict:
    """Load metadata from a meta.json file in the given path."""
    meta_file = os.path.join(path, "meta.json")
    if not os.path.exists(meta_file):
        return {}
    with open(meta_file, "r") as f:
        data = json.load(f)
    return data
