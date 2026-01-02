from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "active"}

def test_generate_data():
    response = client.get("/generate-data")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert 'ds' in data[0] and 'y' in data[0]

def test_train_model():
    response = client.post("/train", json={})
    assert response.status_code == 200
    result = response.json()
    assert result.get("status") == "success"
    assert "model" in result
    assert "run_id" in result
    assert "experiment_id" in result

def test_predict():
    # First, train a model to get a valid run_id
    train_response = client.post("/train", json={})
    assert train_response.status_code == 200
    train_result = train_response.json()
    run_id = train_result.get("run_id")
    
    # Now, make a prediction request
    predict_payload = {
        "model_type": "prophet",
        "run_id": run_id,
        "days": 10,
        "freq": "D"
    }
    predict_response = client.post("/predict", json=predict_payload)
    assert predict_response.status_code == 200
    forecast_data = predict_response.json()
    assert isinstance(forecast_data, list)
    assert len(forecast_data) == 10
    assert 'ds' in forecast_data[0] and 'yhat' in forecast_data[0]