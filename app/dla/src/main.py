from fastapi import FastAPI, Response, status
import uvicorn
from datetime import datetime
import torch

from dla_pipeline_inference import process_documents

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/")
async def hello(response: Response, name: str):
    """Returns a greeting for the name passed in the 'name' parameter"""

    # Handdle empty name case
    if len(name) == 0:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        payload = (
            f"Status code {response.status_code} (incorrect entry): "
            + "The 'name' parameter must have a lenth greater than 0"
        )
    else:
        payload = f"Hello {name}!!"

    return payload


@app.get("/health")
async def get_health():
    return {
        "CONTAINER":"DLA",
        "Current-Time": datetime.now().isoformat()
        }

@app.get("/cuda_check")
async def get_health():
    is_available = torch.cuda.is_available()

    if is_available:
        return f"CUDA Available ({torch.cuda.get_device_name(0)})"
    else:
        return "CUDA is not available"   


@app.get("/dla")
async def get_dla_masks():

    exit_code  = process_documents(full_processing=True,
                                   continue_from_previous=False, model_type='DIT') 
    
    return exit_code


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)