# import external libraries
from fastapi.testclient import TestClient

# import module to test
from backend.app import app

client = TestClient(app)

def test_read_status():
    response = client.get("/welcome")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Fashion Generation API"}
