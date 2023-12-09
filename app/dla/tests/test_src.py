from fastapi.testclient import TestClient

import os
import sys

# So that pytest can find the app folder ######################################
# NOTE: This is ugly and not very robust, but for whatever reason I can't 
# get pytest to find all modules despite having the __init__.py files.

PYTEST_APP_DIR=os.environ.get('PYTEST_APP_DIR')

if PYTEST_APP_DIR==None:
    PYTEST_APP_DIR = os.environ.get("APP_DIR")
    os.environ["PYTEST_APP_DIR"] = PYTEST_APP_DIR

sys.path.append(PYTEST_APP_DIR)

from main import app
###############################################################################


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

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_openapi_json():
    response = client.get("/openapi.json")
    assert response.status_code == 200
