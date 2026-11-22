"""Download the original UCI Online Retail archive without modifying it."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

from retainiq.data.profile import DATASET_URL


def download_source(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(DATASET_URL, timeout=60) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/online_retail.zip"))
    args = parser.parse_args()
    path = download_source(args.output)
    print(f"Downloaded UCI source archive to {path}")


if __name__ == "__main__":
    main()
