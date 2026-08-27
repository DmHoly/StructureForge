import pytest

from structureforge.core.recipe_store import RecipeStore
from structureforge.core.recipes import DepositionMode, DepositionRecipe, EtchMode, EtchRecipe


@pytest.fixture
def store(tmp_path):
    return RecipeStore(tmp_path / "recipes.json")


def test_loading_a_store_with_no_file_yet_is_empty(store):
    library = store.load()
    assert library.deposition == {}
    assert library.etch == {}


def test_upsert_persists_across_store_instances(tmp_path):
    path = tmp_path / "recipes.json"
    RecipeStore(path).upsert_deposition(
        DepositionRecipe(name="Mon depot maison", mode=DepositionMode.conformal, notes="test")
    )

    reloaded = RecipeStore(path).load()
    assert reloaded.deposition["Mon depot maison"].notes == "test"


def test_upsert_replaces_a_custom_recipe_of_the_same_name(store):
    store.upsert_etch(EtchRecipe(name="Ma gravure", mode=EtchMode.isotropic, default_factor=0.5))
    store.upsert_etch(EtchRecipe(name="Ma gravure", mode=EtchMode.isotropic, default_factor=0.9))

    assert store.load().etch["Ma gravure"].default_factor == 0.9


def test_remove_deletes_only_the_named_custom_recipe(store):
    store.upsert_deposition(DepositionRecipe(name="A", mode=DepositionMode.conformal))
    store.upsert_deposition(DepositionRecipe(name="B", mode=DepositionMode.conformal))

    store.remove_deposition("A")

    assert set(store.load().deposition) == {"B"}


def test_combined_with_layers_custom_recipes_over_the_defaults(store, recipes):
    store.upsert_deposition(DepositionRecipe(name="Mon depot maison", mode=DepositionMode.conformal))

    combined = store.combined_with(recipes)

    assert "Mon depot maison" in combined.deposition
    assert "ALD Conformal" in combined.deposition  # the base library is still there


def test_combined_with_lets_a_custom_recipe_override_a_built_in_by_name(store, recipes):
    store.upsert_deposition(DepositionRecipe(name="ALD Conformal", mode=DepositionMode.conformal, notes="overridden"))

    combined = store.combined_with(recipes)

    assert combined.deposition["ALD Conformal"].notes == "overridden"


def test_removing_an_override_reverts_to_the_built_in_on_next_combine(store, recipes):
    store.upsert_deposition(DepositionRecipe(name="ALD Conformal", mode=DepositionMode.conformal, notes="overridden"))
    store.remove_deposition("ALD Conformal")

    combined = store.combined_with(recipes)

    assert combined.deposition["ALD Conformal"].notes != "overridden"
