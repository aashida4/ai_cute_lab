"""Run with: uv run --locked python -m unittest discover -s tests."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import ipywidgets as widgets
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MLPComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "notebooks/02_participant_workshop.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
        cls.code = next(
            "".join(c["source"]) for c in notebook["cells"]
            if c.get("id") == "mlp-comparison-code"
        )

    def setUp(self):
        self.ns = dict(np=np, pd=pd, widgets=widgets, Image=Image,
                       Pipeline=Pipeline, StandardScaler=StandardScaler,
                       LogisticRegression=LogisticRegression,
                       display=lambda *args: None, clear_output=lambda **kwargs: None)
        exec(self.code, self.ns)
        self.x = np.random.default_rng(8).normal(size=(24, 1280))
        self.mask = np.arange(24) >= 20
        self.validation_labels = np.full(24, -1)
        self.validation_labels[self.mask] = [0, 1, 0, 1]
        self.ns.update(holdout_mask=self.mask, holdout_labels=self.validation_labels,
                       record_evaluation=lambda *args: None)

    def fit(self, labels):
        return self.ns["fit_mlp_comparison"](self.x, labels, self.mask, self.validation_labels)

    def test_requires_both_classes(self):
        for labels in (np.full(24, -1), np.zeros(24, dtype=int)):
            with self.assertRaisesRegex(ValueError, "最低1枚"):
                self.fit(labels)

    def test_small_sample_and_holdout(self):
        for count in (1, 5):
            labels = np.full(24, -1)
            labels[:count] = 0
            labels[count:count * 2] = 1
            snapshot = labels.copy()
            # Even accidental training labels on held-out rows must be ignored.
            labels[self.mask] = 1
            snapshot = labels.copy()
            result = self.fit(labels)
            train, check = result["train_ids"], result["check_ids"]
            np.testing.assert_array_equal(check, np.flatnonzero(self.mask))
            self.assertFalse(set(train) & set(check))
            np.testing.assert_array_equal(labels, snapshot)
            for name, pipeline in result["models"].items():
                np.testing.assert_allclose(
                    pipeline["scaler"].mean_, self.x[train].mean(axis=0))
                values = result["scores"][name]
                self.assertEqual(values.shape, (24,))
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(((values >= 0) & (values <= 1)).all())
            self.assertEqual(result["models"]["3層MLP"]["clf"].coefs_[0].shape, (1280, 16))

    def test_callback_preserves_main_state(self):
        labels = np.array([0, 1] * 5 + [-1] * 14)
        main_model = object()
        main_scores = np.full(24, 0.25)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.png"
            Image.new("RGB", (20, 20)).save(path)
            self.ns.update(embeddings=self.x, labels=labels, model=main_model,
                           scores=main_scores, image_path=lambda _: path)
            self.ns["show_mlp_comparison"]()
        self.assertIs(self.ns["model"], main_model)
        self.assertIs(self.ns["scores"], main_scores)
        np.testing.assert_array_equal(self.ns["labels"], labels)
        self.assertFalse(self.ns["mlp_compare_button"].disabled)

    def test_missing_validation_answers(self):
        self.validation_labels[-1] = -1
        with self.assertRaisesRegex(ValueError, "確認用"):
            self.fit(np.array([0, 1] * 12))


if __name__ == "__main__":
    unittest.main()
