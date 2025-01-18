from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io

app = FastAPI()

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"}
)

@app.post("/caption/")
async def caption_image(file: UploadFile = File(...), length: str = Form(...)):
    image = Image.open(io.BytesIO(await file.read()))
    if length == "short":
        caption = model.caption(image, length="short")["caption"]
    elif length == "normal":
        caption = model.caption(image, length="normal")["caption"]
    else:
        return JSONResponse(status_code=400, content={"message": "Invalid length parameter"})

    return {"caption": caption}

@app.post("/visual_query/")
async def visual_query(file: UploadFile = File(...), query: str = Form(...)):
    image = Image.open(io.BytesIO(await file.read()))
    answer = model.query(image, query)["answer"]
    return {"answer": answer}

@app.post("/object_detection/")
async def object_detection(file: UploadFile = File(...), object_type: str = Form(...)):
    image = Image.open(io.BytesIO(await file.read()))
    objects = model.detect(image, object_type)["objects"]
    return {"objects": objects}

#@app.post("/pointing/")
#async def pointing(file: UploadFile = File(...), object_type: str = Form(...)):
#    image = Image.open(io.BytesIO(await file.read()))
#    points = model.point(image, object_type)["points"]
#    return {"points": points}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

