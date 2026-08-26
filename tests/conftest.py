import pytest

from structureforge.core.materials import default_library
from structureforge.core.recipes import default_recipes


@pytest.fixture
def materials():
    return default_library()


@pytest.fixture
def recipes():
    return default_recipes()
