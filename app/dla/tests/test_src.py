from fastapi.testclient import TestClient

# from src import __version__
from src.main import app

"""
STATUS CODE COMMENTS:
`/hello/`: Should produce a status code of **422 Unprocessable Entity**, 
because it is expecting for us to pass the `name` parameter.

`/hello/?name=`: Should produce a status code of **406 Not Acceptable**, 
because a name should be entered so that the page reply makes sense.

`/`: Should produce a status code of **404 Not Found**, 
because the endpoint has not been implemented.
"""

client = TestClient(app)

def test_hello_name_passed():
    response = client.get("/hello", params='name=Carlos')
    assert response.status_code == 200

def test_hello_bad_url():
    response = client.get("/hello")
    assert response.status_code == 422

def test_hello_empty_name_passed():
    response = client.get("/hello", params='name=')
    assert response.status_code == 406

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_json():
    response = client.get("/openapi.json")
    assert response.status_code == 200
