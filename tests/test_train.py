import json

import train


def test_model_is_saved_with_its_metadata(trained_model_dir):
    assert (trained_model_dir / 'rating_model.joblib').exists()
    meta = json.loads((trained_model_dir / 'model_meta.json').read_text())
    assert meta['params'] == train.PARAMS
    assert set(meta['metrics']) >= {'accuracy', 'macro_f1', 'f1_1_star', 'f1_5_star'}
    assert meta['metrics']['macro_f1'] >= 0.5


def test_quality_gate_blocks_a_weak_model(reviews_csv, tmp_path):
    out = tmp_path / 'models'
    _, metrics, passed = train.train(reviews_csv, out, min_macro_f1=1.01,
                                     tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    assert not passed
    # A model that fails the gate must not replace the served one
    assert not out.exists()


def test_training_is_reproducible(reviews_csv, tmp_path):
    runs = [train.train(reviews_csv, tmp_path / f'm{i}', 0.0, f"sqlite:///{tmp_path / 'mlflow.db'}")[1]
            for i in range(2)]
    assert runs[0] == runs[1]
