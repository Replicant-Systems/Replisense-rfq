from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
import shutil
import uvicorn
from file_parser import FileParser
from rfq_agent import RFQFieldGenerator

load_dotenv()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

parser = FileParser()
field_generator = RFQFieldGenerator()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        parsed = parser.parse_file(filepath)
        result = field_generator.generate(
            raw_text=parsed["raw_text"],
            source_file=parsed["source_file"]
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

@app.post("/parse-text/")
async def parse_text(payload: dict):
    try:
        raw_text = payload.get("text", "")
        result = field_generator.generate(raw_text, source_file="email_body")
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )