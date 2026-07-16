import json
import tempfile
import unittest
from pathlib import Path

from bigbrainlayer.pipeline import (
    evaluate_strategies,
    merge_datasets,
    preprocess_bigbrain,
    preprocess_spatial,
    run_pipeline,
)


class PipelineTests(unittest.TestCase):
    def test_merge_and_evaluate(self):
        bigbrain = preprocess_bigbrain(
            [
                {"region": "A", "thickness_mm": "2.0"},
                {"region": "B", "thickness_mm": "3.0"},
                {"region": "C", "thickness_mm": "4.0"},
            ]
        )
        spatial = preprocess_spatial(
            [
                {"region": "A", "expression": "10"},
                {"region": "A", "expression": "12"},
                {"region": "B", "expression": "20"},
                {"region": "C", "expression": "35"},
            ]
        )

        merged = merge_datasets(bigbrain, spatial)
        self.assertEqual(len(merged), 3)

        ranking = evaluate_strategies(merged)
        self.assertEqual(len(ranking), 3)
        self.assertIn("strategy", ranking[0])
        self.assertIn("score", ranking[0])

    def test_pipeline_saves_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bigbrain_path = tmp_path / "bigbrain.csv"
            spatial_path = tmp_path / "spatial.csv"

            bigbrain_path.write_text(
                "region,thickness_mm\nA,2.1\nB,2.9\nC,3.6\n",
                encoding="utf-8",
            )
            spatial_path.write_text(
                "region,expression\nA,8\nA,10\nB,15\nC,25\n",
                encoding="utf-8",
            )

            result = run_pipeline(str(bigbrain_path), str(spatial_path), str(tmp_path / "out"))
            self.assertEqual(result["n_merged_regions"], 3)
            self.assertTrue(Path(result["merged_csv"]).exists())
            self.assertTrue(Path(result["scores_json"]).exists())
            self.assertTrue(Path(result["scores_plot_svg"]).exists())

            payload = json.loads(Path(result["scores_json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 3)


if __name__ == "__main__":
    unittest.main()
