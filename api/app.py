from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
from api.model import get_embedding

app = FastAPI(title="VGG Image Embedding Service")

@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        shutil.copyfileobj(file.file, tmp)
        img_path = tmp.name

    vector = get_embedding(img_path)

    return {
        "model": "vgg16",
        "dimension": len(vector),
        "embedding": vector
    }