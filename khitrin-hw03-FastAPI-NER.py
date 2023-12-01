import functools
from fastapi import FastAPI, HTTPException, status, Response, BackgroundTasks
from fastapi.responses import RedirectResponse
from uuid import uuid4
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder


app = FastAPI(title='HW03 for SWE course UrFU',
              summary='NER extraction from text',
              version="0.0.1")


@functools.cache
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    return pipeline("ner", model=model, tokenizer=tokenizer)


# MVP-решение, в проде необходимо хранение в БД
tasks = {}


class InputText(BaseModel):
    text_for_extraction: str


class NEROutput(BaseModel):
    entity: str
    score: float
    index: int
    word: str
    start: int
    end: int


@app.get("/", include_in_schema=False)
async def redirect_root():
    response = RedirectResponse(url="/docs")
    return response


@app.post("/api/v1/ner/tasks", status_code=status.HTTP_202_ACCEPTED,
          tags=['Create NER Task for provided text'])
async def submit_task(input: InputText, background_tasks: BackgroundTasks):
    task_id = str(uuid4())

    tasks[task_id] = {
        "status": "processing",
        "input": jsonable_encoder(input)
    }

    background_tasks.add_task(extract_ner, task_id)

    return {"task_id": task_id}


async def extract_ner(task_id: str):
    text = tasks[task_id]["input"]["text_for_extraction"]
    result = await run_ner(text)

    json_entities = [
        {
            "entity": entity["entity"],
            "score": float(entity["score"]),
            "index": int(entity["index"]),
            "word": entity["word"],
            "start": int(entity["start"]),
            "end": int(entity["end"])
        }
        for entity in result
    ]

    tasks[task_id]["output"] = json_entities
    tasks[task_id]["status"] = "done"


async def run_ner(text):
    ner_model = load_ner_model()
    prediction = ner_model(text)
    return prediction


@app.get("/api/v1/ner/tasks/{task_id}", tags=['Read Extraction Result for Task'])
async def get_task_status(task_id):
    if task_id not in tasks:
        raise HTTPException(status_code=404)

    if tasks[task_id]["status"] == "processing":
        return Response(status_code=status.HTTP_102_PROCESSING,
                        content={"status": "processing"})

    return tasks[task_id]
