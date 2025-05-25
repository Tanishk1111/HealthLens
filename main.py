import os
import io
import logging
from typing import List, Dict, Any, Optional
import tempfile
import asyncio

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import requests
from dotenv import load_dotenv

# Import our custom modules
from vision_processing.model_2d_handler import Model2DHandler
from vision_processing.model_3d_handler import Model3DHandler
from vision_processing.common import ImageProcessor, ScanTypeValidator
from sonar_integration.client import PerplexityClient

# Configuration
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title="HealthLens Medical Agent API",
    description="AI-powered medical scan analysis with expert consultation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global handlers - will be initialized on startup
model_2d_handler: Optional[Model2DHandler] = None
model_3d_handler: Optional[Model3DHandler] = None
perplexity_client: Optional[PerplexityClient] = None
image_processor: ImageProcessor = ImageProcessor()
scan_validator: ScanTypeValidator = ScanTypeValidator()

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_2d_handler, model_3d_handler, perplexity_client
    
    logger.info("Initializing HealthLens API...")
    
    try:
        # Initialize 2D model handler
        model_2d_handler = Model2DHandler()
        await model_2d_handler.initialize()
        logger.info("2D model handler initialized successfully")
        
        # Initialize 3D model handler
        model_3d_handler = Model3DHandler()
        await model_3d_handler.initialize()
        logger.info("3D model handler initialized successfully")
        
        # Initialize Perplexity client
        if PERPLEXITY_API_KEY:
            perplexity_client = PerplexityClient(PERPLEXITY_API_KEY)
            logger.info("Perplexity client initialized successfully")
        else:
            logger.warning("Perplexity API key not found. Sonar integration will be unavailable.")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down HealthLens API...")
    
    if model_2d_handler:
        await model_2d_handler.cleanup()
    if model_3d_handler:
        await model_3d_handler.cleanup()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HealthLens Medical Agent API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "api_status": "healthy",
        "models": {
            "2d_handler": model_2d_handler is not None and model_2d_handler.is_ready(),
            "3d_handler": model_3d_handler is not None and model_3d_handler.is_ready()
        },
        "services": {
            "perplexity": perplexity_client is not None
        }
    }

@app.get("/supported_scan_types")
async def get_supported_scan_types():
    """Get list of supported scan types"""
    return {
        "supported_scan_types": scan_validator.get_supported_types(),
        "2d_types": scan_validator.get_2d_types(),
        "3d_types": scan_validator.get_3d_types()
    }

@app.post("/analyze_scan/")
async def analyze_scan(
    file: UploadFile = File(..., description="Medical scan file (DICOM, NIfTI, PNG, JPG)"),
    scan_type: str = Form(..., description="Type of scan (e.g., 'xray_chest_2d', 'ct_abdomen_3d')"),
    include_sonar_analysis: bool = Form(True, description="Whether to include Perplexity Sonar analysis"),
    confidence_threshold: float = Form(0.5, description="Minimum confidence threshold for detections")
):
    """
    Main endpoint for analyzing medical scans
    """
    logger.info(f"Received scan analysis request: {scan_type}, file: {file.filename}")
    
    try:
        # Validate scan type
        if not scan_validator.is_valid_scan_type(scan_type):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported scan type: {scan_type}. Use /supported_scan_types to see valid options."
            )
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Read file content
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file provided")
        
        # Determine processing type (2D or 3D)
        is_3d = scan_validator.is_3d_scan_type(scan_type)
        
        # Process the scan
        if is_3d:
            if not model_3d_handler or not model_3d_handler.is_ready():
                raise HTTPException(status_code=503, detail="3D model handler not available")
            
            vision_results = await model_3d_handler.process_scan(
                file_content, 
                file.filename, 
                scan_type,
                confidence_threshold
            )
        else:
            if not model_2d_handler or not model_2d_handler.is_ready():
                raise HTTPException(status_code=503, detail="2D model handler not available")
            
            vision_results = await model_2d_handler.process_scan(
                file_content, 
                scan_type,
                confidence_threshold
            )
        
        # Prepare response
        response_data = {
            "scan_type": scan_type,
            "filename": file.filename,
            "processing_type": "3D" if is_3d else "2D",
            "vision_analysis": {
                "detections": vision_results.get("detections", []),
                "metadata": vision_results.get("metadata", {}),
                "processing_time": vision_results.get("processing_time", 0)
            },
            "sonar_analysis": None
        }
        
        # Add Sonar analysis if requested and available
        if include_sonar_analysis and perplexity_client:
            try:
                sonar_result = await perplexity_client.analyze_findings(
                    vision_results.get("detections", []),
                    scan_type,
                    "3D" if is_3d else "2D"
                )
                response_data["sonar_analysis"] = sonar_result
            except Exception as e:
                logger.warning(f"Sonar analysis failed: {e}")
                response_data["sonar_analysis"] = {
                    "error": "Sonar analysis unavailable",
                    "details": str(e)
                }
        elif include_sonar_analysis and not perplexity_client:
            response_data["sonar_analysis"] = {
                "error": "Perplexity client not configured"
            }
        
        logger.info(f"Analysis completed successfully for {file.filename}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scan analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze_batch/")
async def analyze_batch_scans(
    files: List[UploadFile] = File(...),
    scan_types: List[str] = Form(...),
    include_sonar_analysis: bool = Form(True),
    confidence_threshold: float = Form(0.5)
):
    """
    Batch analysis endpoint for multiple scans
    """
    if len(files) != len(scan_types):
        raise HTTPException(
            status_code=400, 
            detail="Number of files must match number of scan types"
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400, 
            detail="Batch size limited to 10 files"
        )
    
    results = []
    
    for file, scan_type in zip(files, scan_types):
        try:
            # Process each file individually
            result = await analyze_scan(
                file=file,
                scan_type=scan_type,
                include_sonar_analysis=include_sonar_analysis,
                confidence_threshold=confidence_threshold
            )
            results.append({
                "filename": file.filename,
                "status": "success",
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "error"])
    }

@app.get("/model_info")
async def get_model_info():
    """Get information about loaded models"""
    info = {
        "2d_models": {},
        "3d_models": {}
    }
    
    if model_2d_handler and model_2d_handler.is_ready():
        info["2d_models"] = model_2d_handler.get_model_info()
    
    if model_3d_handler and model_3d_handler.is_ready():
        info["3d_models"] = model_3d_handler.get_model_info()
    
    return info

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
