from main import predict
from pydantic import BaseModel

class MockRequest(BaseModel):
    client_id: int

def test_predict_valid_client():
    request = MockRequest(client_id=100007)
    response = predict(request)
    assert response["prediction"] in [0, 1]

def test_predict_invalid_client():
    request = MockRequest(client_id=999999)  # ID inexistant
    try:
        predict(request)
    except Exception as e:
        assert "Client ID not found" in str(e)
