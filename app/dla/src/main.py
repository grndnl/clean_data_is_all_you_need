from fastapi import FastAPI, Response, status
import uvicorn
from datetime import datetime
import torch

from dla_pipeline_inference import process_documents, available_models, script_directory

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Clean Data DLA Endpoint"}


@app.get("/health")
async def get_health():
    return {"CONTAINER": "DLA", "Current-Time": datetime.now().isoformat()}


@app.get("/cuda_check")
async def get_health():
    is_available = torch.cuda.is_available()

    if is_available:
        return f"CUDA Available ({torch.cuda.get_device_name(0)})"
    else:
        return "CUDA is not available"


@app.get("/debug_notes")
async def debug_notes():
    notes_dict = {
        "dla_script_directory": script_directory(),
        "gpu_available": torch.cuda.is_available(),
    }
    return notes_dict


@app.get("/dla")
async def get_dla_masks(
    response: Response, use_cpu: bool = False, model_type: str = "DIT"
):
    response_dict = {
        "opt_use_cpu": use_cpu,
        "opt_model_type": model_type,
        "output_directory": "",
        "text_output_directory": "",
        "exit_code": -1,  # no execution
        "exit_msg": "",
    }

    if model_type not in available_models:
        response.status_code = status.HTTP_406_NOT_ACCEPTABLE
        response_dict["exit_msg"] = (
            f"Status code {response.status_code} (incorrect entry): "
            + f"MODEL_TYPE: {model_type}, not recognized"
        )

        return response_dict

    (
        response_dict["exit_code"],
        response_dict["exit_msg"],
        response_dict["output_directory"],
        response_dict["text_output_directory"],
    ) = process_documents(
        full_inference=True,
        continue_from_previous=False,
        model_type=model_type,
        use_cpu=use_cpu,
    )

    if response_dict["exit_code"] != 0:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    return response_dict


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
