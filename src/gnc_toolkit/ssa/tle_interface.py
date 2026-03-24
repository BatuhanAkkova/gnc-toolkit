"""
TLE Catalog Interface and Utilities.
"""

from ..propagators.sgp4_propagator import Sgp4Propagator


class TLEEntity:
    def __init__(self, name: str, line1: str, line2: str):
        self.name = name.strip()
        self.line1 = line1.rstrip()
        self.line2 = line2.rstrip()
        self.norad_id = self._parse_norad_id()

    def _parse_norad_id(self) -> str:
        # Line 1 columns 3-7 (0-indexed 2 to 6)
        if len(self.line1) > 7:
            return self.line1[2:7].strip()
        return "00000"

    def get_propagator(self) -> Sgp4Propagator:
        """Returns an Sgp4Propagator for this TLE."""
        return Sgp4Propagator(self.line1, self.line2)


class TLECatalog:
    """
    Manages a collection of TLEs for search and lookup.
    """

    def __init__(self):
        self.satellites: list[TLEEntity] = []
        self._by_id: dict[str, TLEEntity] = {}
        self._by_name: dict[str, TLEEntity] = {}

    def add_tle(self, name: str, line1: str, line2: str):
        """Add a single TLE entity."""
        entity = TLEEntity(name, line1, line2)
        self.satellites.append(entity)
        self._by_id[entity.norad_id] = entity
        self._by_name[entity.name.upper()] = entity

    def load_from_txt(self, filepath: str):
        """
        Loads TLEs from a text file.
        Assumes 3-line format (Name, Line 1, Line 2).
        """
        with open(filepath) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        i = 0
        while i < len(lines) - 2:
            # Check if line 1 starts with '1' and line 2 with '2'
            name = lines[i]
            l1 = lines[i + 1]
            l2 = lines[i + 2]

            if l1.startswith("1") and l2.startswith("2"):
                self.add_tle(name, l1, l2)
                i += 3
            else:
                # 2-line format
                if name.startswith("1") and l1.startswith("2"):
                    self.add_tle(f"SAT_{name[2:7]}", name, l1)
                    i += 2
                else:
                    # Skip invalid line
                    i += 1

    def get_by_norad_id(self, norad_id: str) -> TLEEntity | None:
        return self._by_id.get(str(norad_id).strip())

    def get_by_name(self, name: str) -> TLEEntity | None:
        return self._by_name.get(name.strip().upper())

    def list_satellites(self) -> list[str]:
        return [s.name for s in self.satellites]
