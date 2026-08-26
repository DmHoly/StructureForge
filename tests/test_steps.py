import pytest
from pydantic import ValidationError

from structureforge.core.units import Length
from structureforge.process.steps import Lithography, Planarization


def test_planarization_needs_exactly_one_target():
    with pytest.raises(ValidationError, match="exactly one"):
        Planarization(name="CMP")
    with pytest.raises(ValidationError, match="exactly one"):
        Planarization(name="CMP", target_level=Length.nm(10), stop_material="W")

    # each alone is fine
    Planarization(name="CMP", target_level=Length.nm(10))
    Planarization(name="CMP", stop_material="W")


def test_lithography_rejects_a_backwards_opening():
    with pytest.raises(ValidationError, match="not a valid x-range"):
        Lithography(
            name="Masque", resist_material="Photoresist", thickness=Length.nm(5), openings=[(120, 80)]
        )
