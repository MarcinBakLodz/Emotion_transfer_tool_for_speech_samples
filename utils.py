import argparse
import json


def get_parser_from_json(file_path):
    parser = argparse.ArgumentParser(description=f"Parsing arguments from {file_path}")
    args, _ = parser.parse_known_args()
    with open(file_path, 'r') as f:
        config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)
    return args
