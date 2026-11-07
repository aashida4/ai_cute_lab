import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FixedHoldoutTests(unittest.TestCase):
    def test_workflow(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               "notebooks/02_participant_workshop.ipynb").read_text())
        sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        setup = next(s for s in sources if "def make_holdout_mask" in s)
        main = next(s for s in sources if "def choose_next_index" in s)
        variants = next(s for s in sources if "def train_ai_with_strength" in s)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "workshop"
            root.mkdir(parents=True)
            for i in range(40):
                Image.new("RGB", (8, 8)).save(root / f"{i}.png")
            frame = pd.DataFrame({"image_path": [f"{i}.png" for i in range(40)]})
            ns = dict(np=np, pd=pd, widgets=widgets, Image=Image, random=random,
                      metadata=frame, workshop_dir=root,
                      embeddings=np.random.default_rng(42).normal(size=(40, 12)),
                      Pipeline=Pipeline, StandardScaler=StandardScaler,
                      LogisticRegression=LogisticRegression,
                      display=lambda *a: None, clear_output=lambda **k: None)
            exec(setup, ns)
            mask = ns["holdout_mask"].copy()
            reordered = frame.iloc[::-1].reset_index(drop=True)
            reverse_mask = ns["make_holdout_mask"](reordered)
            self.assertEqual(set(frame.loc[mask, "image_path"]),
                             set(reordered.loc[reverse_mask, "image_path"]))
            exec(main, ns)
            self.assertIsNone(ns["choose_next_index"]())
            for i in range(mask.sum()):
                ns["answer_holdout"](i % 2)
            pool = np.flatnonzero(~mask)
            for count in (10, 20):
                ns["labels"][pool[:count]] = np.arange(count) % 2
                ns["train_model"]()
                train = (ns["labels"] != -1) & ~mask
                np.testing.assert_allclose(ns["model"]["scaler"].mean_,
                                           ns["embeddings"][train].mean(axis=0))
                for mode in ("random", "auto", "uncertain"):
                    ns["mode_dropdown"].value = mode
                    for _ in range(10):
                        self.assertFalse(mask[ns["choose_next_index"]()])
            # Deliberately contaminate labels to check the explicit training mask.
            ns["labels"][mask] = 1
            functions = [n for n in ast.parse(variants).body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body=functions, type_ignores=[]), "<variants>", "exec"), ns)
            for strength in (0.1, 1, 1000):
                fitted, _ = ns["train_ai_with_strength"](strength)
                train = (ns["labels"] != -1) & ~mask
                np.testing.assert_allclose(fitted["scaler"].mean_, ns["embeddings"][train].mean(axis=0))
            self.assertEqual([r["学習枚数"] for r in ns["evaluation_history"]], [10, 20])
            self.assertEqual(len({r["確認画像No."] for r in ns["evaluation_history"]}), 1)
            answers = ns["holdout_labels"].copy()
            ns["reset_labels"]()
            np.testing.assert_array_equal(ns["holdout_mask"], mask)
            np.testing.assert_array_equal(ns["holdout_labels"], answers)
            self.assertFalse(ns["evaluation_history"])


if __name__ == "__main__":
    unittest.main()
