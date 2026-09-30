from typer.testing import CliRunner

from evidence_rag.cli import app


def test_cli_help_lists_phase_one_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "index" in result.stdout
    assert "evaluate" in result.stdout
