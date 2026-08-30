import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


class TestFastAPIEndpoints(unittest.TestCase):
    def test_health_check(self):
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_get_chennai_clusters(self):
        response = client.get("/api/v1/clusters")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["city"], "chennai")
        self.assertEqual(len(data["clusters"]), 6)

    def test_predict_real_demand(self):
        payload = {"city": "chennai", "cluster_id": 1, "horizon": 1}
        response = client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["landmark_name"], "t_nagar")
        self.assertGreaterEqual(data["predicted_demand"], 0.0)

    def test_predict_invalid_cluster(self):
        payload = {"city": "chennai", "cluster_id": 99, "horizon": 1}
        response = client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 422)



if __name__ == "__main__":
    unittest.main()
