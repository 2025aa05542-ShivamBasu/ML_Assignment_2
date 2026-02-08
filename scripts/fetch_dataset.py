#!/usr/bin/env python3
"""Download or copy the Obesity dataset into ./data/ with the expected filename.

Usage:
  python scripts/fetch_dataset.py --url <DIRECT_DOWNLOAD_URL>
  python scripts/fetch_dataset.py --local /path/to/ObesityDataSet_raw_and_data_sinthetic.csv

This script avoids extra dependencies by using urllib and shutil.
"""
import argparse
import os
import shutil
from urllib.request import urlopen, Request


DEST_DIR = os.path.join("data")
DEST_NAME = "ObesityDataSet_raw_and_data_sinthetic.csv"


def download_url(url: str, dest_path: str):
    req = Request(url, headers={"User-Agent": "curl/7.64"})
    with urlopen(req) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)


def copy_local(src: str, dest_path: str):
    shutil.copyfile(src, dest_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Direct URL to download the CSV")
    parser.add_argument("--local", help="Path to a local CSV to copy into data/")
    args = parser.parse_args()

    os.makedirs(DEST_DIR, exist_ok=True)
    dest_path = os.path.join(DEST_DIR, DEST_NAME)

    if args.local:
        if not os.path.exists(args.local):
            print(f"Local file not found: {args.local}")
            raise SystemExit(1)
        print(f"Copying {args.local} -> {dest_path}")
        copy_local(args.local, dest_path)
        print("Done.")
        return

    if args.url:
        print(f"Downloading {args.url} -> {dest_path}")
        try:
            download_url(args.url, dest_path)
        except Exception as e:
            print(f"Download failed: {e}")
            raise SystemExit(1)
        print("Done.")
        return

    print("Please provide --url or --local. See script help for usage.")


if __name__ == "__main__":
    main()
