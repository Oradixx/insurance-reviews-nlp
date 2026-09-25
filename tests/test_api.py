import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope='module')
def client(trained_model_dir, monkeypatch_module):
    monkeypatch_module.setenv('MODEL_DIR', str(trained_model_dir))
    import importlib

    import api
    importlib.reload(api)  # MODEL_DIR is read at import time
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope='module')
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_model_info(client):
    assert 'metrics' in client.get('/model').json()


def test_predict_returns_a_rating_and_probabilities(client):
    r = client.post('/predict', json={'text': 'excellent perfect wonderful fast claim'})
    assert r.status_code == 200
    body = r.json()
    assert body['rating'] == 5
    assert set(body['probabilities']) == {'1', '2', '3', '4', '5'}
    assert abs(sum(body['probabilities'].values()) - 1) < 1e-3


def test_batch(client):
    r = client.post('/predict/batch', json={'texts': [{'text': 'worst scandal avoid'}, {'text': 'good helpful fair'}]})
    assert [p['rating'] for p in r.json()] == [1, 4]


@pytest.mark.parametrize('payload', [{}, {'text': ''}, {'text': '   '}, {'text': 'x' * 5001}])
def test_invalid_input_is_rejected(client, payload):
    assert client.post('/predict', json=payload).status_code == 422
