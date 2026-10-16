from ecosystem_evolution import winter_example
from ecosystem_evolution.adaptation_story import show_adaptation_story
from tests.test_simulation import _environment


def test_winter_example_runs_and_reports_a_lineage(capsys) -> None:
    winter_example.main()
    output = capsys.readouterr().out

    assert "Winter forest at 0 C:" in output
    assert "trait match" in output
    assert "One line of descent" in output


def test_lineage_section_is_optional(capsys) -> None:
    favourable = _environment(
        temperature_c=15.0,
        food_availability=0.90,
        water_availability=0.90,
        vegetation_cover=0.90,
        predator_pressure=0.00,
        pollution_pressure=0.00,
        habitat_area_km2=10.0,
    )

    show_adaptation_story("Table only", favourable, runs=5, generations=3, include_lineage=False)
    without_lineage = capsys.readouterr().out
    show_adaptation_story("Table and lineage", favourable, runs=5, generations=3)
    with_lineage = capsys.readouterr().out

    assert "trait match" in without_lineage
    assert "Founder " not in without_lineage
    assert "One line of descent" not in without_lineage
    assert "One line of descent" in with_lineage
