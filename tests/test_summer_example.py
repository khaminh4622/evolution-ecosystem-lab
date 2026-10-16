from ecosystem_evolution import summer_example
from ecosystem_evolution.adaptation_story import show_adaptation_story
from tests.test_simulation import _environment


def test_summer_example_reports_a_warmer_forest(capsys) -> None:
    summer_example.main(include_lineage=False)
    output = capsys.readouterr().out

    assert "Summer forest at 30 C:" in output
    assert "trait match" in output
    assert "One line of descent" not in output


def test_summer_favours_heat_over_insulation(capsys) -> None:
    hot = _environment(
        temperature_c=30.0,
        food_availability=0.70,
        water_availability=0.70,
        vegetation_cover=0.60,
        predator_pressure=0.15,
        pollution_pressure=0.03,
        habitat_area_km2=3.0,
    )
    show_adaptation_story("Hot", hot, runs=5, generations=2, include_lineage=False)
    printed = capsys.readouterr().out
    favoured = printed.splitlines()[3]

    # At 30 C the target fur insulation is low and heat tolerance is high, the
    # reverse of the 0 C winter run.
    assert "fur_ 0.14" in favoured
    assert "heat 0.80" in favoured
