"""Utility helpers for the FarmAnimalsAlarm package.

This file is intentionally lightweight. Move common helpers here when refactoring code
out of `app/` into the package.
"""


def normalize_classifications(classifications):
    """Normalize a variety of classification outputs into a mapping {species: confidence}.

    Accepts:
    - SpeciesNet v4-style {'classes': [...], 'scores': [...]} where class entries are semicolon-delimited strings
    - Dict-like mapping {species: score}
    - List of dicts [{'label': ..., 'confidence': ...}, ...]
    """
    if not classifications:
        return {}

    out = {}
    try:
        if (
            isinstance(classifications, dict)
            and "classes" in classifications
            and "scores" in classifications
        ):
            for class_str, score in zip(
                classifications.get("classes", []), classifications.get("scores", [])
            ):
                parts = class_str.split(";")
                if len(parts) >= 7 and parts[6]:
                    species_name = parts[6].strip()
                elif len(parts) >= 6 and parts[5]:
                    genus = parts[4] if len(parts) > 4 else ""
                    species = parts[5]
                    species_name = f"{genus} {species}".strip() if genus else species
                elif len(parts) >= 2 and parts[1]:
                    species_name = parts[1].strip()
                else:
                    species_name = "Unknown"
                if species_name.lower() not in ["unknown", "blank", ""]:
                    out[species_name.title()] = float(score)
            return out

        if isinstance(classifications, dict):
            for k, v in classifications.items():
                if isinstance(v, (int, float)) and isinstance(k, str):
                    out[k.title()] = float(v)
            return out

        if isinstance(classifications, list):
            for c in classifications:
                if isinstance(c, dict):
                    species = c.get("label", c.get("species"))
                    conf = c.get("confidence", 0.0)
                    if species and isinstance(conf, (int, float)):
                        out[str(species).title()] = float(conf)
            return out
    except Exception:
        return {}

    return out
