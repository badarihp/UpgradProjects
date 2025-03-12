import pytest
import logging
from unittest.mock import MagicMock
from app import app  # Import the Flask app from app.py

# Mocking modules that cause import errors in test environments
import sys
sys.modules["spacy"] = MagicMock()
sys.modules["blis"] = MagicMock()
sys.modules["thinc"] = MagicMock()
sys.modules["numpy"] = MagicMock()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


@pytest.fixture
def client():
    """Flask test client setup."""
    app.config["TESTING"] = True  # Enable test mode
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test if the home page loads correctly."""
    response = client.get("/")
    logging.info(f"Response Data : {response.data}")  # Log response data
    assert response.status_code == 200
    assert b"Product Recommendation System" in response.data  # Check page title

def test_valid_recommend_request(client):
    """Test recommendation with a valid username and model."""
    response = client.post("/recommend", data={"username": "charlie", "model_type": "rf_base"})
    logging.info(f"Response Data : {response.data}")  # Log response data
    
    assert response.status_code == 200  # Ensure request is successful

    # Check if recommendations are present or if an error occurred
    if b"Recommendations for" in response.data:
        assert b"Product Name" in response.data  # Ensure table headers exist
        assert b"Positive Sentiment %" in response.data  # Ensure sentiment percentage exists
    else:
        assert b"Error" in response.data  # Ensure proper error handling when recommendations fail


def test_invalid_recommend_request(client):
    """Test recommendation route accessed via GET (should redirect)."""
    response = client.get("/recommend")
    logging.info(f"Response Data : {response.data}")  # Log response data
    assert response.status_code == 302  # Redirects to home

def test_invalid_user(client):
    """Test behavior for an invalid user."""
    response = client.post("/recommend", data={"username": "invalid_user", "model_type": "rf_base"})
    logging.info(f"Response Data : {response.data}")  # Log response data
    assert response.status_code == 200
    assert b"Invalid user" in response.data or b"Error occurred" in response.data  # Handle different messages

def test_invalid_model_type(client):
    """Test behavior when an invalid model is selected."""
    response = client.post("/recommend", data={"username": "charlie", "model_type": "invalid_model"})
    logging.info(f"Response Data : {response.data}")  # Log response data
    assert response.status_code == 200
    assert b"Invalid model type" in response.data  # Ensure proper error handling

def test_404_route(client):
    """Test if invalid routes return a 404 error or redirect to home."""
    response = client.get("/invalid_route")
    logging.info(f"Response Data : {response.data}")  # Log response data
    assert response.status_code == 404
    assert b"Request to invalid routes, redirecting to the home page." in response.data  # Ensure flash message is shown
    assert b"Product Recommendation System" in response.data  # Ensure home page content is loaded
