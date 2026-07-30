import json
from typing import Any, Dict, List

class MountainParser:
    def parse(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Turn raw data into a list of mountain dictionaries."""
        if raw_data is None:
            raise RuntimeError("No raw data provided. Call fetch() first.")

        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                raw_data = []

        if isinstance(raw_data, dict) and "results" in raw_data:
            return self._parse_sparql_results(raw_data)

        if isinstance(raw_data, list):
            return [item.copy() for item in raw_data if isinstance(item, dict)]

        return []

    def _parse_sparql_results(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        bindings = raw.get("results", {}).get("bindings", [])
        parsed: List[Dict[str, Any]] = []
        for row in bindings:
            name = row.get("mountainLabel", {}).get("value") or row.get("mountain", {}).get("value")
            elevation = row.get("elevation", {}).get("value")
            country = row.get("countryLabel", {}).get("value")
            parsed.append({"name": name, "elevation_m": elevation, "country": country})
        return parsed