from fastapi.testclient import TestClient

from gmoney.api import app


def test_health_endpoints_identify_development_mode() -> None:
    client = TestClient(app)
    assert client.get("/health/live").json() == {"status": "ok"}
    assert client.get("/health/ready").json() == {
        "status": "ready",
        "environment": "development",
        "tenant_mode": "fixed-development",
    }

