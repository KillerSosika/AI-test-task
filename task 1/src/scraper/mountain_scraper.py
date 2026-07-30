from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


class MountainScraper:
    """Download mountain metadata from Wikidata."""

    DEFAULT_ENDPOINT = "https://query.wikidata.org/sparql"

    def __init__(self, endpoint_url: str = DEFAULT_ENDPOINT) -> None:
        self.endpoint_url = endpoint_url

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "MountainNER/1.0",
                "Accept": "application/sparql-results+json",
            }
        )

    def _build_query(self, limit: int, offset: int) -> str:
        return f"""
        SELECT ?mountain ?mountainLabel ?elevation
        WHERE {{
            ?mountain wdt:P31 wd:Q8502.

            OPTIONAL {{
                ?mountain wdt:P2044 ?elevation.
            }}

            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en".
            }}
        }}
        LIMIT {limit}
        OFFSET {offset}
        """

    def _request(self, query: str) -> Optional[Dict]:

        for attempt in range(3):

            try:

                response = self.session.get(
                    self.endpoint_url,
                    params={"query": query},
                    timeout=60,
                )

                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")

                if "json" not in content_type.lower():
                    raise RuntimeError(
                        f"Unexpected content type: {content_type}"
                    )

                return response.json()

            except (
                requests.RequestException,
                requests.exceptions.JSONDecodeError,
                RuntimeError,
            ) as exc:

                print(exc)

                if attempt == 2:
                    print("Skipping this chunk...")
                    return None

                wait = 5 * (attempt + 1)

                print(f"Waiting {wait} seconds...")

                time.sleep(wait)

        return None

    def _parse_item(
        self,
        item: Dict[str, Dict[str, str]],
    ) -> Optional[Dict[str, str]]:

        name = item.get("mountainLabel", {}).get("value", "").strip()

        if not name:
            return None

        if name.startswith("Q"):
            return None

        if not any(c.isalpha() for c in name):
            return None

        elevation = item.get(
            "elevation",
            {},
        ).get(
            "value",
            "Unknown",
        )

        return {
            "name": name,
            "elevation": elevation,
        }

    def fetch_mountains(
        self,
        limit: int = 5000,
        chunk_size: int = 500,
    ) -> List[Dict[str, str]]:

        print(f"Fetching up to {limit} mountains...")

        mountains: List[Dict[str, str]] = []

        seen = set()

        offset = 0

        while offset < limit:

            query = self._build_query(
                min(chunk_size, limit - offset),
                offset,
            )

            data = self._request(query)

            if data is None:
                print("Stopping because Wikidata is unavailable.")
                break

            bindings = data.get(
                "results",
                {},
            ).get(
                "bindings",
                [],
            )

            if not bindings:
                break

            for item in bindings:

                mountain = self._parse_item(item)

                if mountain is None:
                    continue

                if mountain["name"] in seen:
                    continue

                seen.add(mountain["name"])
                mountains.append(mountain)

            offset += chunk_size

            print(
                f"Downloaded {len(mountains)} unique mountains..."
            )

            time.sleep(1)

        mountains.sort(
            key=lambda x: x["name"]
        )

        return mountains

    def save_to_csv(
        self,
        mountains: List[Dict[str, str]],
        filepath: str | Path,
    ) -> None:

        path = Path(filepath)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "elevation",
                ],
            )

            writer.writeheader()
            writer.writerows(mountains)

        print(f"Saved {len(mountains)} mountains to {path}")

    def fetch_and_save(
        self,
        filepath: str | Path,
        limit: int = 5000,
        chunk_size: int = 500,
    ) -> List[Dict[str, str]]:

        mountains = self.fetch_mountains(
            limit=limit,
            chunk_size=chunk_size,
        )

        if mountains:
            self.save_to_csv(
                mountains,
                filepath,
            )

        print(
            f"Finished with {len(mountains)} unique mountains."
        )

        return mountains