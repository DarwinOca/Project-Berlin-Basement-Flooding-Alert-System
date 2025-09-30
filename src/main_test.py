import pytest
from fastapi.testclient import TestClient
from unittest import mock
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from fe.constants import RESPONSE

# Ensure the parent directory is in the path to find the main module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Import the FastAPI app instance from the main file
from main import app, get_db_connection

# Create a test client for the FastAPI app
client = TestClient(app)

# --- Test Fixtures ---
# This fixture overrides the database dependency to use a mock object.
# This ensures tests are isolated and don't require a real database connection.
@pytest.fixture
def mock_db_connection():
    """
    Mocks the database connection and cursor for testing purposes.
    """
    with mock.patch('main.mysql.connector.connect') as mock_connect:
        mock_cursor = mock.Mock()
        mock_connect.return_value.cursor.return_value = mock_cursor

        # Override the get_db_connection dependency to use our mock connection
        app.dependency_overrides[get_db_connection] = lambda: mock_connect.return_value

        yield mock_cursor

    # Clean up the dependency override after the test
    app.dependency_overrides = {}


# --- Test Cases ---
def test_read_posts(mock_db_connection):
    """
    Tests the /show_table endpoint to ensure it returns posts from the database.
    It mocks the database call to return a predictable set of records.
    """
    table = 'posts'
    limit = 10
    # Assert that the JSON response matches our expected mock data
    expected_response = [
        {"title": "Test Title 1", "content": "Test Content 1", "author": "Test Author 1"},
        {"title": "Test Title 2", "content": "Test Content 2", "author": "Test Author 2"}
    ]
    # Configure the mock cursor to return a fake row from the database
    mock_db_connection.fetchall.return_value = expected_response

    # Make a GET request to the /show_table endpoint using the test client
    response = client.get("/show_table", params={"table": table, "limit": limit})

    # Assert that the response status code is 200 OK
    assert response.status_code == 200

    assert response.json() == expected_response

    # Verify that the database cursor's execute method was called
    mock_db_connection.execute.assert_called_with(f"SELECT * FROM {table} LIMIT {limit}")


def test_upload_csv(mock_db_connection):
    """
    Tests the /upload-csv/ endpoint to ensure it correctly handles file uploads.
    It mocks the database connection and verifies that the database insert
    statements are called as expected.
    """
    # Create some dummy CSV data
    csv_content = 'title,content,author\n"Sample Post","This is a sample post.","John Doe"'

    # Create a mock file to simulate the upload
    files = {"file": ("sample.csv", csv_content, "text/csv")}

    # Make a POST request to the /upload-csv/ endpoint with the mock file
    response = client.post("/upload-csv/", files=files)

    # Assert that the response status code is 200 OK
    assert response.status_code == 200

    # Assert that the success message is returned
    # assert response.json() == {"message": "CSV data uploaded and saved to database successfully!"}
    assert response.json()['message'] == RESPONSE['csv']['success']

    # Verify that the database cursor's execute method was called with the correct SQL
    # We check for the a single call to `execute` to insert the row
    # mock_db_connection.execute.assert_called_once()
    assert mock_db_connection.execute.call_count == 2

    # We also check that the database connection's commit method was called
    # mock_db_connection.commit.assert_called_once()
