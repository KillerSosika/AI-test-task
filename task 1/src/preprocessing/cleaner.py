from typing import Any, Dict, List

class MountainCleaner:
    def clean(self, mountains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize fields and convert to the final CSV-friendly shape."""
        cleaned: List[Dict[str, Any]] = []
        
        for m in mountains:
            name = m.get("name") or m.get("title") or ""
            elevation = m.get("elevation_m") or m.get("elevation") or None
            
            if isinstance(elevation, dict):
                elevation = elevation.get("value")

            try:
                if elevation is not None and elevation != "":
                    elevation = int(str(elevation).replace(",", "").strip())
                else:
                    elevation = None
            except (ValueError, TypeError):
                elevation = None

            country = m.get("country") or m.get("countryLabel") or ""
            if isinstance(country, list):
                country = ", ".join(str(item) for item in country if item)

            cleaned.append(
                {
                    "name": str(name).strip(),
                    "elevation_m": elevation,
                    "country": str(country).strip() or None,
                }
            )

        return cleaned