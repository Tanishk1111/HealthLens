# vision_processing/model_3d_handler.py
import os
import time
import logging
import subprocess # For running TotalSegmentator
import tempfile
import shutil
from typing import Dict, List, Any, Optional

from .common import BaseImageProcessor, BaseModelOutputProcessor # Assuming common.py

logger = logging.getLogger(__name__)

class Model3DHandler:
    """Handler for 3D medical imaging models, primarily TotalSegmentator."""
    
    def __init__(self):
        self.is_initialized = False
        self.totalsegmentator_available = False
        self.image_processor = BaseImageProcessor()
        self.output_processor = BaseModelOutputProcessor()

        self.scan_type_to_task = {
            'ct_full_body_segmentation_3d': 'totalsegmentator_ct_fast',
            'mri_brain_segmentation_3d': 'totalsegmentator_mri_fast', # Assuming you might use its MRI capabilities
            # Add other 3D scan_type mappings if you use MONAI 3D via this handler or other 3D models
        }
            
    async def initialize(self):
        if self.is_initialized:
            logger.info("3D model handler already initialized.")
            return

        logger.info("Initializing 3D model handler...")
        # Check if TotalSegmentator command is available
        try:
            # Use --help as a lightweight way to check if the command exists and is runnable
            result = subprocess.run(["TotalSegmentator", "--help"], capture_output=True, text=True, check=False)
            if result.returncode == 0 and "TotalSegmentator" in result.stdout:
                self.totalsegmentator_available = True
                logger.info("TotalSegmentator command found and seems operational.")
            else:
                logger.warning(f"TotalSegmentator command check failed or not found. Stdout: {result.stdout}, Stderr: {result.stderr}")
                self.totalsegmentator_available = False
        except FileNotFoundError:
            logger.error("TotalSegmentator command not found. Please ensure it's installed and in PATH.")
            self.totalsegmentator_available = False
        except Exception as e:
            logger.error(f"Error checking TotalSegmentator availability: {e}", exc_info=True)
            self.totalsegmentator_available = False
            
        self.is_initialized = True # Mark as initialized even if TS is not found, to avoid re-checking
        logger.info(f"3D model handler initialization complete. TotalSegmentator available: {self.totalsegmentator_available}")

    def is_ready(self) -> bool:
        # For TotalSegmentator, readiness depends on it being installed and found.
        return self.is_initialized and self.totalsegmentator_available
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "TotalSegmentator": {
                "available": self.totalsegmentator_available,
                "description": "Segments 100+ structures in CT, and some in MRI. Uses nnU-Net.",
                "tasks_supported": list(self.scan_type_to_task.keys())
            }
            # Add info for other 3D models if managed by this handler
        }
        
    async def process_scan(
        self, 
        input_scan_path: str, # Expects a file path for TotalSegmentator
        scan_type: str,
        # filename_for_metadata: str = "uploaded_3d_scan.tmp", # Less relevant as TS operates on paths
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.is_ready(): # Checks if TS is available
            logger.error("3D Model Handler (TotalSegmentator) not ready.")
            raise RuntimeError("TotalSegmentator not available or handler not initialized.")

        task = self.scan_type_to_task.get(scan_type)
        if not task:
            logger.error(f"No 3D task configured for scan_type '{scan_type}'.")
            raise ValueError(f"Unsupported 3D scan_type: {scan_type}")
        
        # Create a temporary directory for TotalSegmentator output
        # This directory will be cleaned up.
        temp_output_dir = tempfile.mkdtemp(prefix="totalseg_out_")
        logger.info(f"Created temporary output directory for TotalSegmentator: {temp_output_dir}")

        results: Dict[str, Any] = {}
        try:
            if task.startswith('totalsegmentator_'):
                # For TotalSegmentator, success means output files were generated.
                # The "detections" will be a list of segmented structures.
                ts_output = await self._run_totalsegmentator(
                    input_scan_path, 
                    temp_output_dir, 
                    fast_mode= ("_fast" in task),
                    is_mri= ("_mri_" in task)
                )
                
                if ts_output.get("success"):
                    segmented_files = [
                        os.path.join(temp_output_dir, f) 
                        for f in os.listdir(temp_output_dir) 
                        if f.endswith(".nii.gz") and f != "total.nii.gz" # Exclude the combined mask
                    ]
                    segmented_structures_names = [
                        os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]
                        for f in segmented_files
                    ]

                    results['detections'] = [ # Format somewhat like 2D handler for consistency
                        self.output_processor.format_detection_result(
                            class_name=name,
                            confidence=1.0, # Segmentation is binary (present or not)
                            segmentation_info={"mask_file": file_path} # Path to individual mask
                        ) for name, file_path in zip(segmented_structures_names, segmented_files)
                    ]
                    # For main.py, you might want to pass temp_output_dir so it can decide to keep/zip/delete
                    results['raw_output_info'] = {
                        "output_directory_temp": temp_output_dir, # Pass this back for potential cleanup or zipping
                        "segmented_structures_list": segmented_structures_names,
                        "log": ts_output.get("log")
                    }
                else:
                    results['error'] = ts_output.get("error", "TotalSegmentator execution failed.")
                    results['raw_output_info'] = {"log": ts_output.get("log")}
            else:
                raise ValueError(f"Unsupported 3D task type: {task}")

        except Exception as e:
            logger.error(f"Error processing 3D scan ({scan_type}) with task {task}: {e}", exc_info=True)
            results['error'] = f"Processing failed: {str(e)}"
        
        # Metadata section
        processing_time = time.time() - start_time
        metadata = {
            "input_scan_path_provided": input_scan_path, # Can't get full metadata if only path
            "model_used": "TotalSegmentator",
            "model_task": task,
            "scan_type_processed": scan_type,
            "processing_time_seconds": round(processing_time, 3)
        }
        results['metadata'] = metadata

        # Note: `temp_output_dir` and its contents should be managed by the caller (e.g., FastAPI endpoint)
        # The caller can decide to zip it, serve files from it, or delete it.
        # If you want this handler to delete, add cleanup here or in a separate method.
        # For now, we return the path.
        
        return results

    async def _run_totalsegmentator(
        self, input_path: str, output_dir: str, fast_mode: bool = True, is_mri: bool = False
    ) -> Dict[str, Any]:
        if not self.totalsegmentator_available:
            return {"success": False, "error": "TotalSegmentator command is not available."}

        command = ["TotalSegmentator", "-i", input_path, "-o", output_dir]
        if fast_mode:
            command.append("--fast")
        if is_mri:
            command.append("--ml") # Use --ml for multi-label (full body) MRI segmentation if that's the target
            # Or specify task for MRI if TS supports task-based MRI segmentation now
            # command.extend(["--task", "lung_vessels"]) # Example, check TS docs for MRI tasks

        log_output = ""
        try:
            logger.info(f"Running TotalSegmentator command: {' '.join(command)}")
            # Use asyncio.create_subprocess_exec for non-blocking in async context
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            log_output += f"STDOUT:\n{stdout.decode(errors='ignore')}\n"
            log_output += f"STDERR:\n{stderr.decode(errors='ignore')}\n"

            if process.returncode == 0:
                logger.info(f"TotalSegmentator completed successfully for {input_path}.")
                # Verify output files were created
                if not os.listdir(output_dir): # Basic check
                     return {"success": False, "error": "TotalSegmentator ran but produced no output files.", "log": log_output}
                return {"success": True, "output_dir": output_dir, "log": log_output}
            else:
                error_msg = f"TotalSegmentator failed with return code {process.returncode}."
                logger.error(error_msg + f"\n{log_output}")
                return {"success": False, "error": error_msg, "log": log_output}
        except FileNotFoundError:
            logger.error("TotalSegmentator command not found during execution attempt.")
            return {"success": False, "error": "TotalSegmentator command not found.", "log": "FileNotFoundError"}
        except Exception as e:
            logger.error(f"Exception running TotalSegmentator: {e}", exc_info=True)
            return {"success": False, "error": str(e), "log": log_output + f"\nException: {str(e)}"}

    async def cleanup_temp_dir(self, dir_path: str):
        """Utility to cleanup temporary directories created by this handler if caller doesn't."""
        if os.path.exists(dir_path) and dir_path.startswith(tempfile.gettempdir()): # Safety check
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Successfully cleaned up temporary directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory {dir_path}: {e}")
        else:
            logger.warning(f"Skipping cleanup of non-existent or non-temporary path: {dir_path}")

    async def cleanup(self):
        logger.info("Cleaning up 3D model handler...")
        # No specific models to clear from memory for TotalSegmentator as it's CLI
        self.is_initialized = False
        logger.info("3D model handler cleaned up.")

import asyncio # Required for create_subprocess_exec
