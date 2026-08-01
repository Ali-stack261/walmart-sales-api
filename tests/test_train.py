from pathlib import Path

from src.train import save_training_summary


def test_save_training_summary_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "training_summary.json"

    summary = {
        "best_model": "XGBoost",
        "best_rmse": 71577.19,
        "best_mae": 46892.2,
        "best_r2": 0.9826,
    }

    save_training_summary(summary, str(output_path))

    assert output_path.exists()
    saved = output_path.read_text(encoding="utf-8")
    assert "XGBoost" in saved
    assert "71577.19" in saved
