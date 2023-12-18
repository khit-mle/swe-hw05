from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_redirects_to_docs():
    response = client.get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_extract_org_entities():
    text_for_extrct = "Google and YouTube are the two most visited websites followed by Facebook."
    response = client.post("/api/v1/ner/tasks",
                           json={"text_for_extraction": text_for_extrct})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['status'] == 'done'
    assert json_data['input']['text_for_extraction'] == text_for_extrct
    assert json_data['output'][0]['entity'] == 'B-ORG'
    assert json_data['output'][0]['word'] == 'Google'
    assert json_data['output'][1]['entity'] == 'B-ORG'
    assert json_data['output'][1]['word'] == 'YouTube'
    assert json_data['output'][2]['entity'] == 'B-ORG'
    assert json_data['output'][2]['word'] == 'Facebook'


def test_extract_loc_entities():
    text_for_extrct = "Dublin is the capital of Ireland which is in Europe."
    response = client.post("/api/v1/ner/tasks",
                           json={"text_for_extraction": text_for_extrct})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['status'] == 'done'
    assert json_data['input']['text_for_extraction'] == text_for_extrct
    assert json_data['output'][0]['entity'] == 'B-LOC'
    assert json_data['output'][0]['word'] == 'Dublin'
    assert json_data['output'][1]['entity'] == 'B-LOC'
    assert json_data['output'][1]['word'] == 'Ireland'
    assert json_data['output'][2]['entity'] == 'B-LOC'
    assert json_data['output'][2]['word'] == 'Europe'
  
