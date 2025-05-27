# main.py
import os
import logging
import tempfile
import shutil
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import your handlers and clients
from vision_processing.model_2d_handler import Model2DHandler
from vision_processing.model_3d_handler import Model3DHandler
from vision_processing.monai_model_handler import MonaiModelHandler
from sonar_integration.client import SonarClient
from vision_processing.common import BaseModelOutputProcessor # For summarizing

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Handler Instances ---
# These will be initialized during app lifespan
model_2d_manager: Optional[Model2DHandler] = None
model_3d_manager: Optional[Model3DHandler] = None
monai_manager: Optional[MonaiModelHandler] = None
sonar_client: Optional[SonarClient] = None
output_processor = BaseModelOutputProcessor() # For text summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_2d_manager, model_3d_manager, monai_manager, sonar_client
    logger.info("Application startup: Initializing services...")
    
    model_2d_manager = Model2DHandler()
    await model_2d_manager.initialize()
    
    model_3d_manager = Model3DHandler()
    await model_3d_manager.initialize()
    
    monai_manager = MonaiModelHandler() # Pass base_bundle_dir if needed from .env
    await monai_manager.initialize()
    
    sonar_client = SonarClient() # API key loaded from .env by its constructor
    
    logger.info("All services initialized.")
    yield
    logger.info("Application shutdown: Cleaning up services...")
    if model_2d_manager: await model_2d_manager.cleanup()
    if model_3d_manager: await model_3d_manager.cleanup() # Implement cleanup if needed
    if monai_manager: await monai_manager.cleanup() # Implement cleanup
    logger.info("Services cleaned up.")

app = FastAPI(title="HealthLens Medical Imaging Analysis API", version="0.1.0", lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or restrict to your frontend's origin in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- Helper for Temporary File Management ---
async def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    try:
        # Create a temporary file with the same suffix as the original file
        suffix = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await upload_file.read()
            tmp_file.write(content)
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file to temporary location: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

# --- API Endpoints ---
@app.get("/health", summary="Check API Health")
async def health_check():
    return {
        "status": "ok",
        "services": {
            "model_2d_handler_ready": model_2d_manager.is_ready() if model_2d_manager else False,
            "model_3d_handler_ready": model_3d_manager.is_ready() if model_3d_manager else False,
            "monai_model_handler_ready": monai_manager.is_ready() if monai_manager else False,
            "sonar_client_configured": bool(sonar_client and sonar_client.api_key)
        }
    }

@app.get("/model_info", summary="Get Information About Loaded Models")
async def get_all_model_info():
    info = {}
    if model_2d_manager: info["2d_models"] = model_2d_manager.get_model_info()
    if model_3d_manager: info["3d_models_totalsegmentator"] = model_3d_manager.get_model_info()
    if monai_manager: info["monai_bundles"] = monai_manager.get_model_info()
    return info

@app.post("/analyze_scan/", summary="Analyze a Medical Scan")
async def analyze_scan_endpoint(
    file: UploadFile = File(..., description="The medical scan image file (e.g., PNG, JPG, NIfTI)."),
    scan_type: str = Form(..., description="Type of scan and analysis desired. Examples: 'xray_chest_pathology_2d', 'ct_full_body_segmentation_3d', 'ct_spleen_segmentation_monai_3d', 'mri_brain_tumor_segmentation_monai_3d'."),
    confidence_threshold: float = Form(0.1, ge=0.0, le=1.0, description="Confidence threshold for filtering 2D model detections."),
    # include_sonar_analysis: bool = Form(True, description="Whether to include analysis from Perplexity Sonar."), # Removed as per simplified flow
    # sonar_query_type: str = Form("professional", description="'professional' or 'patient' for Sonar query style."), # Removed
):
    logger.info(f"Received /analyze_scan/ request for scan_type: {scan_type}, filename: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided for uploaded file.")

    temp_file_path: Optional[str] = None
    temp_output_dirs_to_clean: List[str] = [] # Keep track of temp dirs created by models

    try:
        # Save uploaded file to a temporary path for models that need it
        # For models that accept bytes (like TorchXRayVision), we can pass bytes directly.
        file_content_bytes = await file.read() # Read once
        await file.seek(0) # Reset pointer if needed by other parts or for saving
        
        # Some models (TotalSegmentator, MONAI) need file paths.
        # Only create temp file if it's for a model known to require a path.
        needs_file_path = "segmentation_3d" in scan_type or "monai_3d" in scan_type
        if needs_file_path:
            temp_file_path = await save_upload_file_to_temp(file) # Pass original UploadFile object
            logger.info(f"Uploaded file saved temporarily to: {temp_file_path}")


        vision_analysis_results: Dict[str, Any] = {}

        # --- Route to appropriate handler based on scan_type ---
        if scan_type == 'xray_chest_pathology_2d':
            if not model_2d_manager or not model_2d_manager.is_ready():
                raise HTTPException(status_code=503, detail="2D model service (TorchXRayVision) not ready.")
            vision_analysis_results = await model_2d_manager.process_scan(
                image_bytes=file_content_bytes, # Pass bytes directly
                scan_type=scan_type,
                filename_for_metadata=file.filename,
                confidence_threshold=confidence_threshold
            )
        
        elif scan_type == 'ct_full_body_segmentation_3d' or scan_type == 'mri_brain_segmentation_3d': # TotalSegmentator
            if not model_3d_manager or not model_3d_manager.is_ready():
                raise HTTPException(status_code=503, detail="3D model service (TotalSegmentator) not ready.")
            if not temp_file_path: raise HTTPException(status_code=500, detail="Temp file path missing for 3D scan.")
            
            vision_analysis_results = await model_3d_manager.process_scan(
                input_scan_path=temp_file_path,
                scan_type=scan_type,
            )
            if vision_analysis_results.get("raw_output_info", {}).get("output_directory_temp"):
                temp_output_dirs_to_clean.append(vision_analysis_results["raw_output_info"]["output_directory_temp"])
        
        elif scan_type == 'ct_spleen_segmentation_monai_3d' or scan_type == 'mri_brain_tumor_segmentation_monai_3d': # MONAI
            if not monai_manager or not monai_manager.is_ready():
                raise HTTPException(status_code=503, detail="MONAI model service not ready.")
            if not temp_file_path: raise HTTPException(status_code=500, detail="Temp file path missing for MONAI scan.")

            vision_analysis_results = await monai_manager.process_scan(
                input_scan_path=temp_file_path,
                scan_type=scan_type,
                filename_for_metadata=file.filename,
            )
            if vision_analysis_results.get("raw_output_info", {}).get("temp_output_dir_bundle"):
                temp_output_dirs_to_clean.append(vision_analysis_results["raw_output_info"]["temp_output_dir_bundle"])
        else:
            logger.warning(f"Unsupported or unknown scan_type received: {scan_type}")
            raise HTTPException(status_code=400, detail=f"Unsupported or unknown scan_type: {scan_type}")

        # --- Sonar Integration ---
        sonar_explanation = "Sonar analysis not performed for this request."
        sonar_query_sent = "N/A"
        if sonar_client and sonar_client.api_key and "error" not in vision_analysis_results: # Only query if vision was OK
            sonar_query_sent = sonar_client.prepare_sonar_query_from_vision_results(
                vision_results=vision_analysis_results, 
                scan_type=scan_type,
                query_type="professional" # Default to professional
            )
            sonar_response = await sonar_client.query_sonar(sonar_query_sent) # Make it async if SonarClient.query_sonar is async
            if "error" in sonar_response:
                sonar_explanation = f"Sonar Error: {sonar_response['error']}"
                if sonar_response.get('details'):
                     sonar_explanation += f" Details: {sonar_response['details']}"
            else:
                sonar_explanation = sonar_response.get("explanation", "No explanation from Sonar.")
        elif not (sonar_client and sonar_client.api_key):
            sonar_explanation = "Sonar client not configured (API key missing)."
        elif "error" in vision_analysis_results:
            sonar_explanation = f"Sonar analysis skipped due to vision processing error: {vision_analysis_results['error']}"


        return {
            "scan_type_processed": scan_type,
            "original_filename": file.filename,
            "vision_analysis": vision_analysis_results,
            "sonar_query_sent_to_llm": sonar_query_sent,
            "sonar_llm_explanation": sonar_explanation
        }

    finally:
        # --- Cleanup Temporary Files and Directories ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"Cleaned up temporary input file: {temp_file_path}")
            except Exception as e_clean:
                logger.error(f"Error cleaning up temporary input file {temp_file_path}: {e_clean}")
        
        for temp_dir in temp_output_dirs_to_clean:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary output directory: {temp_dir}")
                except Exception as e_clean_dir:
                    logger.error(f"Error cleaning up temporary output directory {temp_dir}: {e_clean_dir}")

