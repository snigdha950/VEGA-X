from pathlib import Path
import json

import pandas as pd
from PIL import Image

from visualize import run_visualization


ROOT = Path(__file__).resolve().parents[1]


def test_focused_dashboard_generates_expected_outputs(tmp_path):
    outputs = run_visualization(
        file=ROOT / "data/tsrd/config_103.h5",
        slots=30,
        bands=20,
        seed=4,
        output_dir=tmp_path,
        save_gif=False,
        show=False,
    )
    names = {path.name for path in outputs}
    assert names == {
        "policy_comparison.csv",
        "run_summary.json",
        "rafts_live_dashboard.png",
        "rafts_evidence_summary.png",
    }
    comparison = pd.read_csv(tmp_path / "policy_comparison.csv")
    assert comparison["policy_key"].tolist() == [
        "sequential",
        "random",
        "coverage_sweep",
        "rafts_cusum_ucb",
    ]
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_summary"].startswith("FROZEN HELD-OUT")
    assert "dashboard_bottom_row" not in summary
    assert summary["receiver_model"]["dwell_slots"] == 1
    assert summary["receiver_model"]["sensitivity_status"].startswith("configured")
    assert summary["policies"]["rafts_cusum_ucb"]["fallback_count"] == 0
    assert (tmp_path / "rafts_live_dashboard.png").stat().st_size > 10_000
    assert (tmp_path / "rafts_evidence_summary.png").stat().st_size > 10_000
    for name in ("rafts_live_dashboard.png", "rafts_evidence_summary.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()
