from pathlib import Path

data_dir = Path(__file__).resolve().parents[3] / "data"
small_data_dir = data_dir / "ModelNet10_PLY"
big_data_dir = data_dir / "ModelNet40_PLY"

trained_dir = Path(__file__).resolve().parents[3] / "trained"
