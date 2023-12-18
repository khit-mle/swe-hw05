from fastapi.testclient import TestClient
from khitrin_hw03_FastAPI_NER import app


client = TestClient(app)


def test_extract_org_entities():
    text_for_extrct = "Google and YouTube are the two most visited websites followed by Facebook."
    create_task_response = client.post("/api/v1/ner/tasks",
                           json={"text_for_extraction": text_for_extrct})
    create_task_json_data = create_task_response.json()
    assert create_task_response.status_code == 202
    task_id = create_task_json_data['task_id']
    read_result_response = client.get(f"/api/v1/ner/tasks/{task_id}")
    read_result_response_json_data = read_result_response.json()
    assert read_result_response.status_code == 200
    assert read_result_response_json_data['status'] == 'done'
    assert read_result_response_json_data['input']['text_for_extraction'] == text_for_extrct
    assert read_result_response_json_data['output'][0]['entity'] == 'B-ORG'
    assert read_result_response_json_data['output'][0]['word'] == 'Google'
    assert read_result_response_json_data['output'][1]['entity'] == 'B-ORG'
    assert read_result_response_json_data['output'][1]['word'] == 'YouTube'
    assert read_result_response_json_data['output'][2]['entity'] == 'B-ORG'
    assert read_result_response_json_data['output'][2]['word'] == 'Facebook'


def test_extract_loc_entities():
    text_for_extrct = "Dublin is the capital of Ireland which is in Europe."
    create_task_response = client.post("/api/v1/ner/tasks",
                           json={"text_for_extraction": text_for_extrct})
    create_task_json_data = create_task_response.json()
    assert create_task_response.status_code == 202
    task_id = create_task_json_data['task_id']
    read_result_response = client.get(f"/api/v1/ner/tasks/{task_id}")
    read_result_response_json_data = read_result_response.json()
    assert read_result_response.status_code == 200
    assert read_result_response_json_data['status'] == 'done'
    assert read_result_response_json_data['input']['text_for_extraction'] == text_for_extrct
    assert read_result_response_json_data['output'][0]['entity'] == 'B-LOC'
    assert read_result_response_json_data['output'][0]['word'] == 'Dublin'
    assert read_result_response_json_data['output'][1]['entity'] == 'B-LOC'
    assert read_result_response_json_data['output'][1]['word'] == 'Ireland'
    assert read_result_response_json_data['output'][2]['entity'] == 'B-LOC'
    assert read_result_response_json_data['output'][2]['word'] == 'Europe'  
