import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scraper.mountain_scraper import MountainScraper


def main():
    scraper = MountainScraper()

    scraper.fetch_and_save(
        filepath="data/raw/mountains.csv",
        limit=5000,
        chunk_size=500,
    )


if __name__ == "__main__":
    main()