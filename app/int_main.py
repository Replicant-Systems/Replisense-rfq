# integrated_main.py - Updated main.py with quote comparison features
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pathlib import Path
import os
import aiofiles
import aiofiles.os
import uvicorn
import logging
from contextlib import asynccontextmanager
import time
from typing import Dict, Any, Optional, List
import tempfile
import uuid
from file_parser import FileParser, FileParsingError
from rfq_agent import RFQFieldGenerator
from quote_comparison_agent import QuoteComparisonAgent, DemoDataGenerator

load_dotenv()


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
STATIC_DIR = Path("./static")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# LLM Configuration
LLM_CONFIG = {
    "model": "llama3-70b-8192",
    "api_key": os.getenv("GROQ_API_KEY"),
    "base_url": "https://api.groq.com/openai/v1",
    "api_type": "openai",
    "temperature": 0.1,
    "max_tokens": 1200,
    "cache_seed": 42,
    "timeout": 30,
}

# Global instances
parser: Optional[FileParser] = None
field_generator: Optional[RFQFieldGenerator] = None
quote_agent: Optional[QuoteComparisonAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global parser, field_generator, quote_agent
    
    logger.info("Starting RFQ Processing & Quote Comparison API...")
    
    # Initialize services
    try:
        parser = FileParser(max_file_size_mb=MAX_FILE_SIZE_MB)
        field_generator = RFQFieldGenerator()
        quote_agent = QuoteComparisonAgent(LLM_CONFIG)
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        raise
    
    yield  # Application runs
    
    # Cleanup
    logger.info("Shutting down RFQ Processing & Quote Comparison API...")

# Create FastAPI app
app = FastAPI(
    title="RFQ Processing & Quote Comparison API",
    description="Complete API for processing RFQ documents and comparing vendor quotes with AI agents",
    version="2.1.0",
    lifespan=lifespan
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files for demo dashboard
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.2f}s")
        
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"❌ {request.method} {request.url.path} - Error: {str(e)} - Duration: {process_time:.2f}s")
        raise

# Dependencies
async def get_parser() -> FileParser:
    if parser is None:
        raise HTTPException(status_code=503, detail="File parser service not initialized")
    return parser

async def get_field_generator() -> RFQFieldGenerator:
    if field_generator is None:
        raise HTTPException(status_code=503, detail="RFQ field generator service not initialized")
    return field_generator

async def get_quote_agent() -> QuoteComparisonAgent:
    if quote_agent is None:
        raise HTTPException(status_code=503, detail="Quote comparison agent not initialized")
    return quote_agent

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
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def validate_file_size(file: UploadFile) -> bool:
    if hasattr(file, 'size') and file.size:
        return file.size <= MAX_FILE_SIZE_MB * 1024 * 1024
    return True

async def save_upload_file(file: UploadFile, destination: Path) -> None:
    try:
        async with aiofiles.open(destination, 'wb') as f:
            while chunk := await file.read(8192):
                await f.write(chunk)
    except Exception as e:
        if destination.exists():
            await aiofiles.os.remove(destination)
        raise e

async def cleanup_file(filepath: Path) -> None:
    try:
        if filepath.exists():
            await aiofiles.os.remove(filepath)
    except Exception as e:
        logger.warning(f"Failed to cleanup file {filepath}: {str(e)}")

# === MAIN ROUTES ===

@app.get("/")
async def root():
    """Health check endpoint"""
    return StandardResponse.success(
        data={
            "status": "healthy", 
            "version": "2.1.0",
            "services": ["rfq_processing", "quote_comparison"],
            "demo_url": "/demo"
        },
        message="RFQ Processing & Quote Comparison API is running"
    )

@app.get("/demo")
async def serve_demo():
    """Serve the demo dashboard"""
    try:
        return FileResponse("demo_dashboard.html", media_type="text/html")
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={"error": "Demo dashboard not found", "details": str(e)}
        )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "api": "healthy",
        "file_parser": "healthy" if parser else "unhealthy",
        "field_generator": "healthy" if field_generator else "unhealthy",
        "quote_agent": "healthy" if quote_agent else "unhealthy",
        "temp_dir": str(TEMP_DIR),
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "supported_extensions": list(ALLOWED_EXTENSIONS),
        "llm_config": {
            "model": LLM_CONFIG.get("model"),
            "api_configured": bool(LLM_CONFIG.get("api_key"))
        }
    }
    
    services_healthy = [
        health_status["file_parser"],
        health_status["field_generator"], 
        health_status["quote_agent"]
    ]
    all_healthy = all(status == "healthy" for status in services_healthy)
    
    return JSONResponse(
        status_code=200 if all_healthy else 503,
        content=StandardResponse.success(health_status, "Health check completed")
    )

# === RFQ PROCESSING ROUTES (Original) ===

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    file_parser: FileParser = Depends(get_parser),
    rfq_generator: RFQFieldGenerator = Depends(get_field_generator)
):
    """Upload and process a file to extract RFQ data"""
    
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
    
    unique_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_filename = f"{unique_id}{file_extension}"
    temp_filepath = TEMP_DIR / temp_filename
    
    try:
        await save_upload_file(file, temp_filepath)
        logger.info(f"📁 Saved uploaded file: {temp_filepath}")
        
        try:
            parsed_result = await file_parser.parse_file_async(str(temp_filepath))
        except FileParsingError as e:
            logger.warning(f"File parsing failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=422, detail=f"File parsing failed: {str(e)}")
        
        try:
            if hasattr(rfq_generator, 'generate_async'):
                rfq_result = await rfq_generator.generate_async(
                    raw_text=parsed_result["raw_text"],
                    source_file=file.filename
                )
            else:
                rfq_result = rfq_generator.generate(
                    raw_text=parsed_result["raw_text"],
                    source_file=file.filename
                )
        except Exception as e:
            logger.error(f"RFQ generation failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"RFQ processing failed: {str(e)}")
        
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
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        await cleanup_file(temp_filepath)

@app.post("/parse-text/")
async def parse_text(
    request: Request,
    rfq_generator: RFQFieldGenerator = Depends(get_field_generator)
):
    """Process raw text input to extract RFQ data"""
    
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {str(e)}")
    
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")
    
    raw_text = payload.get("text", "")
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="Text field is required and cannot be empty")
    
    source_file = payload.get("source_file", "direct_text_input")
    
    try:
        if hasattr(rfq_generator, 'generate_async'):
            result = await rfq_generator.generate_async(raw_text, source_file)
        else:
            result = rfq_generator.generate(raw_text, source_file)
    except Exception as e:
        logger.error(f"RFQ generation failed for text input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"RFQ processing failed: {str(e)}")
    
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

# === QUOTE COMPARISON ROUTES ===

@app.post("/quotes/rfq/{rfq_id}/add-quote/")
async def add_vendor_quote(
    rfq_id: str,
    quote_data: Dict[str, Any],
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Add a vendor quote to an RFQ for comparison"""
    try:
        quote_id = await agent.add_vendor_quote(rfq_id, quote_data)
        
        return StandardResponse.success(
            data={
                "quote_id": quote_id,
                "rfq_id": rfq_id,
                "vendor_name": quote_data.get("vendor_name", "Unknown"),
                "total_quotes_for_rfq": len(agent.quote_storage.get(rfq_id, []))
            },
            message=f"Quote added successfully from {quote_data.get('vendor_name', 'Unknown')}"
        )
        
    except Exception as e:
        logger.error(f"Failed to add vendor quote: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/quotes/rfq/{rfq_id}/compare/")
async def compare_quotes(
    rfq_id: str,
    rfq_title: str = "",
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Compare all quotes for an RFQ and generate recommendations"""
    try:
        comparison = await agent.compare_quotes(rfq_id, rfq_title)
        
        return StandardResponse.success(
            data={
                "comparison_id": comparison.comparison_id,
                "rfq_id": comparison.rfq_id,
                "vendor_count": len(comparison.vendor_quotes),
                "winner_quote_id": comparison.winner_quote_id,
                "total_savings_potential": comparison.total_savings_potential,
                "recommendations": comparison.recommendations,
                "comparison_matrix": comparison.comparison_matrix,
                "created_at": comparison.created_at.isoformat()
            },
            message="Quote comparison completed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Quote comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/quotes/rfq/{rfq_id}/quotes/")
async def get_rfq_quotes(
    rfq_id: str,
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Get all quotes for a specific RFQ"""
    try:
        quotes = agent.quote_storage.get(rfq_id, [])
        
        quotes_data = []
        for quote in quotes:
            quotes_data.append({
                "quote_id": quote.quote_id,
                "vendor_name": quote.vendor_name,
                "vendor_email": quote.vendor_email,
                "total_quote_value": quote.total_quote_value,
                "currency": quote.currency,
                "submission_date": quote.submission_date.isoformat(),
                "line_items_count": len(quote.line_items),
                "vendor_rating": quote.vendor_rating,
                "status": quote.status.value,
                "delivery_terms": quote.delivery_terms,
                "payment_terms": quote.payment_terms
            })
        
        return StandardResponse.success(
            data={
                "rfq_id": rfq_id,
                "quotes": quotes_data,
                "total_quotes": len(quotes_data)
            },
            message=f"Retrieved {len(quotes_data)} quotes for RFQ {rfq_id}"
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve quotes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quotes/demo/setup/")
async def setup_demo_data(
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Setup demo data for client presentation"""
    try:
        sample_rfq = DemoDataGenerator.generate_sample_rfq()
        rfq_id = sample_rfq["rfq_id"]
        
        sample_quotes = DemoDataGenerator.generate_sample_quotes(sample_rfq)
        quote_ids = []
        
        for quote_data in sample_quotes:
            quote_id = await agent.add_vendor_quote(rfq_id, quote_data)
            quote_ids.append(quote_id)
        
        comparison = await agent.compare_quotes(rfq_id, sample_rfq["title"])
        
        return StandardResponse.success(
            data={
                "demo_setup": True,
                "rfq_id": rfq_id,
                "rfq_title": sample_rfq["title"],
                "quotes_added": len(quote_ids),
                "quote_ids": quote_ids,
                "comparison_id": comparison.comparison_id,
                "winner_vendor": next(
                    (q["vendor_name"] for q in comparison.vendor_quotes if q["quote_id"] == comparison.winner_quote_id),
                    "Unknown"
                ),
                "savings_potential": comparison.total_savings_potential,
                "recommendations_count": len(comparison.recommendations)
            },
            message="Demo data setup completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Demo setup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Demo setup failed: {str(e)}")

@app.get("/quotes/demo/dashboard/{rfq_id}/")
async def get_demo_dashboard(
    rfq_id: str,
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Get comprehensive dashboard data for demo presentation"""
    try:
        quotes = agent.quote_storage.get(rfq_id, [])
        if not quotes:
            raise HTTPException(status_code=404, detail="No quotes found for this RFQ")
        
        comparison = None
        for comp in agent.comparison_storage.values():
            if comp.rfq_id == rfq_id:
                comparison = comp
                break
        
        if not comparison:
            raise HTTPException(status_code=404, detail="No comparison found for this RFQ")
        
        dashboard_data = {
            "rfq_info": {
                "rfq_id": rfq_id,
                "title": comparison.rfq_title,
                "total_quotes_received": len(quotes),
                "comparison_date": comparison.created_at.isoformat(),
                "status": "completed" if comparison.analysis_complete else "in_progress"
            },
            "quotes_overview": [
                {
                    "vendor_name": quote.vendor_name,
                    "total_value": quote.total_quote_value,
                    "currency": quote.currency,
                    "vendor_rating": quote.vendor_rating,
                    "line_items": len(quote.line_items),
                    "delivery_terms": quote.delivery_terms,
                    "is_winner": quote.quote_id == comparison.winner_quote_id
                }
                for quote in quotes
            ],
            "cost_analysis": {
                "lowest_quote": min(quote.total_quote_value for quote in quotes),
                "highest_quote": max(quote.total_quote_value for quote in quotes),
                "average_quote": sum(quote.total_quote_value for quote in quotes) / len(quotes),
                "potential_savings": comparison.total_savings_potential,
                "savings_percentage": (comparison.total_savings_potential / max(quote.total_quote_value for quote in quotes)) * 100 if quotes else 0
            },
            "recommendations": comparison.recommendations,
            "comparison_matrix": comparison.comparison_matrix,
            "winner_analysis": {
                "winner_quote_id": comparison.winner_quote_id,
                "winner_vendor": next(
                    (quote.vendor_name for quote in quotes if quote.quote_id == comparison.winner_quote_id),
                    "Unknown"
                ),
                "winner_value": next(
                    (quote.total_quote_value for quote in quotes if quote.quote_id == comparison.winner_quote_id),
                    0
                )
            }
        }
        
        return StandardResponse.success(
            data=dashboard_data,
            message="Dashboard data retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quotes/comparison/{comparison_id}/")
async def get_comparison_details(
    comparison_id: str,
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Get detailed comparison results"""
    try:
        if comparison_id not in agent.comparison_storage:
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        comparison = agent.comparison_storage[comparison_id]
        
        return StandardResponse.success(
            data={
                "comparison_id": comparison.comparison_id,
                "rfq_id": comparison.rfq_id,
                "rfq_title": comparison.rfq_title,
                "created_at": comparison.created_at.isoformat(),
                "vendor_quotes": comparison.vendor_quotes,
                "comparison_matrix": comparison.comparison_matrix,
                "recommendations": comparison.recommendations,
                "winner_quote_id": comparison.winner_quote_id,
                "total_savings_potential": comparison.total_savings_potential,
                "analysis_complete": comparison.analysis_complete
            },
            message="Comparison details retrieved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comparison details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quotes/analytics/")
async def get_analytics_overview(
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Get overall analytics for all RFQs and comparisons"""
    try:
        total_rfqs = len(agent.quote_storage)
        total_comparisons = len(agent.comparison_storage)
        total_quotes = sum(len(quotes) for quotes in agent.quote_storage.values())
        
        total_savings = sum(comp.total_savings_potential for comp in agent.comparison_storage.values())
        avg_savings = total_savings / total_comparisons if total_comparisons > 0 else 0
        
        vendor_stats = {}
        for quotes in agent.quote_storage.values():
            for quote in quotes:
                if quote.vendor_name not in vendor_stats:
                    vendor_stats[quote.vendor_name] = {
                        "quotes_submitted": 0,
                        "total_value_quoted": 0,
                        "average_rating": 0,
                        "wins": 0
                    }
                
                vendor_stats[quote.vendor_name]["quotes_submitted"] += 1
                vendor_stats[quote.vendor_name]["total_value_quoted"] += quote.total_quote_value
                vendor_stats[quote.vendor_name]["average_rating"] = quote.vendor_rating
        
        for comp in agent.comparison_storage.values():
            if comp.winner_quote_id:
                for quotes in agent.quote_storage.values():
                    winner_quote = next((q for q in quotes if q.quote_id == comp.winner_quote_id), None)
                    if winner_quote:
                        vendor_stats[winner_quote.vendor_name]["wins"] += 1
                        break
        
        analytics_data = {
            "overview": {
                "total_rfqs": total_rfqs,
                "total_quotes": total_quotes,
                "total_comparisons": total_comparisons,
                "average_quotes_per_rfq": total_quotes / total_rfqs if total_rfqs > 0 else 0,
                "total_savings_identified": total_savings,
                "average_savings_per_rfq": avg_savings
            },
            "vendor_performance": [
                {
                    "vendor_name": vendor,
                    "quotes_submitted": stats["quotes_submitted"],
                    "average_quote_value": stats["total_value_quoted"] / stats["quotes_submitted"],
                    "win_rate": (stats["wins"] / stats["quotes_submitted"]) * 100,
                    "vendor_rating": stats["average_rating"]
                }
                for vendor, stats in vendor_stats.items()
            ],
            "recent_activity": [
                {
                    "comparison_id": comp.comparison_id,
                    "rfq_id": comp.rfq_id,
                    "rfq_title": comp.rfq_title,
                    "created_at": comp.created_at.isoformat(),
                    "vendor_count": len(comp.vendor_quotes),
                    "savings_potential": comp.total_savings_potential
                }
                for comp in sorted(agent.comparison_storage.values(), key=lambda x: x.created_at, reverse=True)[:10]
            ]
        }
        
        return StandardResponse.success(
            data=analytics_data,
            message="Analytics overview retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quotes/bulk/add-quotes/")
async def bulk_add_quotes(
    quotes_data: List[Dict[str, Any]],
    agent: QuoteComparisonAgent = Depends(get_quote_agent)
):
    """Add multiple vendor quotes at once"""
    try:
        results = []
        
        for quote_data in quotes_data:
            rfq_id = quote_data.get("rfq_id")
            if not rfq_id:
                results.append({"error": "Missing rfq_id", "quote_data": quote_data})
                continue
            
            try:
                quote_id = await agent.add_vendor_quote(rfq_id, quote_data)
                results.append({
                    "success": True,
                    "quote_id": quote_id,
                    "rfq_id": rfq_id,
                    "vendor_name": quote_data.get("vendor_name", "Unknown")
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "rfq_id": rfq_id,
                    "vendor_name": quote_data.get("vendor_name", "Unknown")
                })
        
        successful_adds = len([r for r in results if r.get("success")])
        
        return StandardResponse.success(
            data={
                "total_processed": len(quotes_data),
                "successful_adds": successful_adds,
                "failed_adds": len(quotes_data) - successful_adds,
                "results": results
            },
            message=f"Bulk add completed: {successful_adds}/{len(quotes_data)} successful"
        )
        
    except Exception as e:
        logger.error(f"Bulk add quotes failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# === WORKFLOW ENDPOINTS ===

@app.post("/workflow/complete-rfq/")
async def complete_rfq_workflow(
    file: UploadFile = File(...),
    file_parser: FileParser = Depends(get_parser),
    rfq_generator: RFQFieldGenerator = Depends(get_field_generator)
):
    """Complete workflow: Upload RFQ -> Extract data -> Return RFQ ID for quote comparison"""
    
    # Use existing upload logic
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    unique_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_filename = f"{unique_id}{file_extension}"
    temp_filepath = TEMP_DIR / temp_filename
    
    try:
        await save_upload_file(file, temp_filepath)
        
        parsed_result = await file_parser.parse_file_async(str(temp_filepath))
        
        rfq_result = await rfq_generator.generate_async(
            raw_text=parsed_result["raw_text"],
            source_file=file.filename
        )
        
        # Generate RFQ ID if not present
        rfq_id = rfq_result.get('rfq_id') or f"RFQ-{unique_id[:8]}"
        rfq_result['rfq_id'] = rfq_id
        
        return StandardResponse.success(
            data={
                "rfq_data": rfq_result,
                "rfq_id": rfq_id,
                "next_steps": {
                    "add_quotes_endpoint": f"/quotes/rfq/{rfq_id}/add-quote/",
                    "compare_quotes_endpoint": f"/quotes/rfq/{rfq_id}/compare/",
                    "demo_dashboard_url": f"/quotes/demo/dashboard/{rfq_id}/"
                }
            },
            message=f"RFQ processed successfully. Ready for quote comparison."
        )
        
    except Exception as e:
        logger.error(f"Complete RFQ workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        await cleanup_file(temp_filepath)

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
            "features": [
                "RFQ document processing with AI field extraction",
                "Multi-vendor quote comparison and analysis",
                "AI-powered procurement recommendations",
                "Real-time cost savings identification",
                "Automated compliance scoring",
                "Interactive demo dashboard"
            ],
            "api_endpoints": {
                "rfq_processing": ["/upload/", "/parse-text/", "/workflow/complete-rfq/"],
                "quote_comparison": ["/quotes/rfq/{rfq_id}/add-quote/", "/quotes/rfq/{rfq_id}/compare/"],
                "demo": ["/quotes/demo/setup/", "/quotes/demo/dashboard/{rfq_id}/", "/demo"]
            }
        },
        message="Supported formats and features retrieved"
    )

# === ERROR HANDLERS ===

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
    
    logger.info(f"Starting integrated RFQ & Quote Comparison server with config: {config}")
    logger.info("Available endpoints:")
    logger.info("  - RFQ Processing: /upload/, /parse-text/")
    logger.info("  - Quote Comparison: /quotes/*")
    logger.info("  - Demo Dashboard: /demo")
    logger.info("  - Health Check: /health")
    
    uvicorn.run(
        "int_main:app",
        **config
    )