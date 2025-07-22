# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import aiofiles
import aiofiles.os
import uvicorn
import logging
from contextlib import asynccontextmanager
import time
from typing import Dict, Any, Optional
import tempfile
from pathlib import Path

# Import our optimized modules
from app.file_parser import FileParser, FileParsingError
from app.rfq_agent import RFQFieldGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.xlsx', '.xls', '.docx', '.csv', '.json'}
UPLOAD_DIR = Path("./uploads")
TEMP_DIR = Path(tempfile.gettempdir()) / "rfq_processing"

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Global instances
parser: Optional[FileParser] = None
field_generator: Optional[RFQFieldGenerator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global parser, field_generator
    
    logger.info("Starting RFQ Processing API...")
    
    # Initialize services
    try:
        parser = FileParser(max_file_size_mb=MAX_FILE_SIZE_MB)
        field_generator = RFQFieldGenerator()
        logger.info("✅ Services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    yield  # Application runs
    
    # Cleanup
    logger.info("Shutting down RFQ Processing API...")

# Create FastAPI app
app = FastAPI(
    title="RFQ Processing API",
    description="API for processing RFQ documents and extracting structured data",
    version="2.0.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure properly in production

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request
    logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.2f}s")
        
        # Add process time to response headers
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - Error: {str(e)} - Duration: {process_time:.2f}s")
        raise

# Dependency for getting services
async def get_parser() -> FileParser:
    if parser is None:
        raise HTTPException(status_code=503, detail="File parser service not initialized")
    return parser

async def get_field_generator() -> RFQFieldGenerator:
    if field_generator is None:
        raise HTTPException(status_code=503, detail="RFQ field generator service not initialized")
    return field_generator

# Response models
class StandardResponse:
    @staticmethod
    def success(data: Any, message: str = "Operation successful") -> Dict[str, Any]:
        return {
            "success": True,
            "data": data,
            "message": message,
            "timestamp": time.time()
        }
    
    @staticmethod
    def error(error: str, details: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "details": details,
            "timestamp": time.time()
        }

# Utility functions
def validate_file_type(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def validate_file_size(file: UploadFile) -> bool:
    """Check if file size is within limits"""
    if hasattr(file, 'size') and file.size:
        return file.size <= MAX_FILE_SIZE_MB * 1024 * 1024
    return True  # If size unknown, let parser handle it

async def save_upload_file(file: UploadFile, destination: Path) -> None:
    """Safely save uploaded file"""
    try:
        async with aiofiles.open(destination, 'wb') as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                await f.write(chunk)
    except Exception as e:
        # Clean up partial file if error occurs
        if destination.exists():
            await aiofiles.os.remove(destination)
        raise e

async def cleanup_file(filepath: Path) -> None:
    """Safely remove file"""
    try:
        if filepath.exists():
            await aiofiles.os.remove(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return StandardResponse.success(
        data={"status": "healthy", "version": "2.0.0"},
        message="RFQ Processing API is running"
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "api": "healthy",
        "file_parser": "healthy" if parser else "unhealthy",
        "field_generator": "healthy" if field_generator else "unhealthy",
        "temp_dir": str(TEMP_DIR),
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "supported_extensions": list(ALLOWED_EXTENSIONS)
    }
    
    all_healthy = all(status == "healthy" for status in [health_status["api"], health_status["file_parser"], health_status["field_generator"]])
    
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content=StandardResponse.success(health_status, "Health check completed")
    )

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    file_parser: FileParser = Depends(get_parser),
    rfq_generator: RFQFieldGenerator = Depends(get_field_generator)
):
    """Upload and process a file"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    if not validate_file_size(file):
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )
    
    # Create unique temporary file path
    import uuid
    unique_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_filename = f"{unique_id}{file_extension}"
    temp_filepath = TEMP_DIR / temp_filename
    
    try:
        # Save uploaded file
        await save_upload_file(file, temp_filepath)
        logger.info(f"📁 Saved uploaded file: {temp_filepath}")
        
        # Parse file
        try:
            parsed_result = await file_parser.parse_file_async(str(temp_filepath))
        except FileParsingError as e:
            logger.warning(f"File parsing failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=422, detail=f"File parsing failed: {str(e)}")
        
        # Generate RFQ fields
        try:
            if hasattr(rfq_generator, 'generate_async'):
                rfq_result = await rfq_generator.generate_async(
                    raw_text=parsed_result["raw_text"],
                    source_file=file.filename
                )
            else:
                # Fallback to sync method
                rfq_result = rfq_generator.generate(
                    raw_text=parsed_result["raw_text"],
                    source_file=file.filename
                )
        except Exception as e:
            logger.error(f"RFQ generation failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RFQ processing failed: {str(e)}")
        
        # Add parsing metadata to result
        rfq_result.update({
            "parsing_info": {
                "original_filename": file.filename,
                "file_size_bytes": parsed_result.get("file_size", 0),
                "parsing_method": parsed_result.get("parsing_method", "unknown"),
                "text_length": len(parsed_result["raw_text"])
            }
        })
        
        return StandardResponse.success(
            data=rfq_result,
            message=f"Successfully processed {file.filename}"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Always cleanup temporary file
        await cleanup_file(temp_filepath)

@app.post("/parse-text/")
async def parse_text(
    request: Request,
    rfq_generator: RFQFieldGenerator = Depends(get_field_generator)
):
    """Process raw text input"""
    
    try:
        # Parse JSON payload
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
        
        # Validate payload
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Payload must be a JSON object")
        
        raw_text = payload.get("text", "")
        if not raw_text or not raw_text.strip():
            raise HTTPException(status_code=400, detail="Text field is required and cannot be empty")
        
        # Optional parameters
        source_file = payload.get("source_file", "direct_text_input")
        
        # Generate RFQ fields
        try:
            if hasattr(rfq_generator, 'generate_async'):
                result = await rfq_generator.generate_async(raw_text, source_file)
            else:
                # Fallback to sync method
                result = rfq_generator.generate(raw_text, source_file)
        except Exception as e:
            logger.error(f"RFQ generation failed for text input: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RFQ processing failed: {str(e)}")
        
        # Add input metadata
        result.update({
            "parsing_info": {
                "input_type": "direct_text",
                "text_length": len(raw_text),
                "source_file": source_file
            }
        })
        
        return StandardResponse.success(
            data=result,
            message="Successfully processed text input"
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error processing text input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/supported-formats/")
async def get_supported_formats():
    """Get list of supported file formats"""
    formats_info = {
        ".txt": "Plain text files",
        ".pdf": "PDF documents (with table support)",
        ".xlsx": "Excel spreadsheets (newer format)",
        ".xls": "Excel spreadsheets (legacy format)",
        ".docx": "Microsoft Word documents (with table support)",
        ".csv": "Comma-separated values",
        ".json": "JSON data files"
    }
    
    return StandardResponse.success(
        data={
            "supported_extensions": list(ALLOWED_EXTENSIONS),
            "format_descriptions": formats_info,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "recommendations": [
                "For best results with PDFs, ensure text is selectable (not scanned images)",
                "Excel files will be limited to first 1000 rows per sheet",
                "Word documents will extract both text and table content",
                "Large files may take longer to process"
            ]
        },
        message="Supported file formats retrieved"
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    return JSONResponse(
        status_code=exc.status_code,
        content=StandardResponse.error(
            error=exc.detail,
            details=f"{request.method} {request.url.path}"
        )
    )

@app.exception_handler(FileParsingError)
async def file_parsing_exception_handler(request: Request, exc: FileParsingError):
    """Handle file parsing exceptions"""
    logger.warning(f"File parsing error on {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content=StandardResponse.error(
            error="File parsing failed",
            details=str(exc)
        )
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=StandardResponse.error(
            error="Internal server error",
            details="An unexpected error occurred. Please check the logs."
        )
    )

if __name__ == "__main__":
    # Development server configuration
    config = {
        "host": "0.0.0.0",
        "port": int(os.getenv("PORT", 8000)),
        "reload": os.getenv("ENVIRONMENT", "development") == "development",
        "log_level": os.getenv("LOG_LEVEL", "info").lower(),
        "access_log": True
    }
    
    logger.info(f"Starting server with config: {config}")
    
    uvicorn.run(
        "main:app",
        **config
    )