from ecosystem_evolution import desert_example


def test_desert_example_runs_and_reports_a_lineage(capsys) -> None:
    desert_example.main()
    output = capsys.readouterr().out

    assert "Desert at 30 C:" in output
    assert "trait match" in output
    assert "One line of descent" in output
