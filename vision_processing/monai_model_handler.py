# vision_processing/monai_model_handler.py
import os
import time
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Optional

from .common import BaseImageProcessor, BaseModelOutputProcessor # Assuming common.py

try:
    import torch
    from monai.bundle import download, ConfigWorkflow
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logging.warning("MONAI or PyTorch not available. MONAI model handler will be non-functional.")

logger = logging.getLogger(__name__)

class MonaiModelHandler:
    """Handler for MONAI Bundles."""
    
    def __init__(self, base_bundle_dir: Optional[str] = None):
        self.base_bundle_dir = base_bundle_dir or os.getenv("MONAI_BUNDLES_BASE_DIR", "./monai_models_downloaded")
        os.makedirs(self.base_bundle_dir, exist_ok=True)
        
        self.loaded_bundle_runners: Dict[str, ConfigWorkflow] = {} # Cache for loaded ConfigWorkflow runners
        self.model_configs: Dict[str, Any] = {} # Store basic info about bundles
        self.is_initialized = False
        self.device = None

        self.image_processor = BaseImageProcessor()
        self.output_processor = BaseModelOutputProcessor()

        # Define which scan_types map to which MONAI bundle and task
        self.scan_type_to_bundle_task = {
            'ct_spleen_segmentation_monai_3d': {
                "bundle_name": "spleen_ct_segmentation", 
                "task_method_name": "run_spleen_segmentation_ct",
                "output_structure_name": "spleen"
            },
            'mri_brain_tumor_segmentation_monai_3d': {
                "bundle_name": "brats_mri_segmentation", 
                "task_method_name": "run_brats_mri_segmentation",
                "output_structure_name": ["enhancing_tumor", "tumor_core", "whole_tumor"] # Multi-label
            }
        }
            
    async def initialize(self):
        if self.is_initialized:
            logger.info("MONAI model handler already initialized.")
            return

        logger.info("Initializing MONAI model handler...")
        if MONAI_AVAILABLE and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using device: cuda for MONAI models.")
        elif MONAI_AVAILABLE:
            self.device = "cpu"
            logger.info("Using device: cpu for MONAI models.")
        else:
            self.device = "cpu" # Fallback, but MONAI won't work
            logger.warning("MONAI not available, MONAI models will fail.")

        # Pre-downloading can be done here, or on-demand in process_scan
        # For hackathon, on-demand is fine. If you want to pre-download:
        # for config in self.scan_type_to_bundle_task.values():
        #     await self._ensure_bundle_downloaded(config["bundle_name"])
            
        self.is_initialized = True
        logger.info("MONAI model handler initialization complete.")

    async def _ensure_bundle_downloaded(self, bundle_name: str, source: str = "monaihosting") -> str:
        bundle_root = os.path.join(self.base_bundle_dir, bundle_name)
        # Check if a key config file exists as a proxy for "downloaded"
        # metadata.json or an inference config are good candidates
        key_config_file = os.path.join(bundle_root, "configs", "inference.json") 
        if not os.path.exists(key_config_file):
            alt_key_config_file = os.path.join(bundle_root, "configs", "eval.json") # Some bundles use eval.json
            if not os.path.exists(alt_key_config_file):
                 key_config_file = os.path.join(bundle_root, "metadata.json") # Fallback to metadata
                 if not os.path.exists(key_config_file):
                    logger.info(f"Bundle '{bundle_name}' key config not found at {bundle_root}. Attempting download...")
                    try:
                        download(
                            name=bundle_name,
                            bundle_dir=self.base_bundle_dir,
                            source=source,
                            progress=True
                        )
                        logger.info(f"Bundle '{bundle_name}' downloaded successfully to {bundle_root}")
                        # Store basic config after download
                        self.model_configs[bundle_name] = {"path": bundle_root, "source": source, "type": "monai_bundle"}
                    except Exception as e:
                        logger.error(f"Failed to download MONAI bundle '{bundle_name}': {e}", exc_info=True)
                        raise # Re-raise to be handled by the caller
                 else:
                    logger.info(f"Bundle '{bundle_name}' (metadata only) found at {bundle_root}")
            else:
                 logger.info(f"Bundle '{bundle_name}' (eval config) found at {bundle_root}")

        else:
            logger.info(f"Bundle '{bundle_name}' (inference config) already available at {bundle_root}")
        
        if bundle_name not in self.model_configs and os.path.exists(bundle_root): # If downloaded but not in config
             self.model_configs[bundle_name] = {"path": bundle_root, "source": source, "type": "monai_bundle"}
        return bundle_root

    def is_ready(self) -> bool:
        return self.is_initialized and MONAI_AVAILABLE
    
    def get_model_info(self) -> Dict[str, Any]:
        info = {}
        for scan_type, config in self.scan_type_to_bundle_task.items():
            bundle_name = config["bundle_name"]
            bundle_path = os.path.join(self.base_bundle_dir, bundle_name)
            info[scan_type] = {
                'type': 'monai_bundle',
                'bundle_name': bundle_name,
                'task': config.get("output_structure_name", "segmentation/classification"),
                'is_downloaded': os.path.exists(os.path.join(bundle_path, "metadata.json")) # Simple check
            }
        return info
        
    async def process_scan(
        self, 
        input_scan_path: str, # Expects a file path for MONAI bundles
        scan_type: str,
        filename_for_metadata: str = "uploaded_monai_scan.tmp",
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.is_ready():
            logger.error("MONAI Model Handler not ready (MONAI or PyTorch might be missing).")
            raise RuntimeError("MONAI Model Handler not ready.")

        task_config = self.scan_type_to_bundle_task.get(scan_type)
        if not task_config:
            logger.error(f"No MONAI task configured for scan_type '{scan_type}'.")
            raise ValueError(f"Unsupported MONAI scan_type: {scan_type}")
        
        bundle_name = task_config["bundle_name"]
        results: Dict[str, Any] = {}
        
        # Create a unique temporary output directory for this run's bundle output
        # This helps avoid conflicts if multiple requests use the same bundle.
        temp_bundle_output_dir = tempfile.mkdtemp(prefix=f"monai_{bundle_name}_out_")
        logger.info(f"Created temp output dir for MONAI bundle {bundle_name}: {temp_bundle_output_dir}")

        try:
            bundle_root = await self._ensure_bundle_downloaded(bundle_name)
            
            # Determine config file path (inference.json usually, sometimes eval.json)
            config_filename = "inference.json"
            inference_config_path = os.path.join(bundle_root, "configs", config_filename)
            if not os.path.exists(inference_config_path):
                config_filename = "eval.json" # Try eval.json as a fallback
                inference_config_path = os.path.join(bundle_root, "configs", config_filename)
                if not os.path.exists(inference_config_path):
                    msg = f"Neither inference.json nor eval.json found for bundle '{bundle_name}' in {os.path.join(bundle_root, 'configs')}"
                    logger.error(msg)
                    raise FileNotFoundError(msg)
            
            # Define output path for the prediction mask
            # The actual output filename might be defined within the bundle, this is a guess
            output_pred_filename = f"{task_config.get('output_structure_name', 'prediction')}_seg.nii.gz"
            if isinstance(task_config.get('output_structure_name'), list): # For multi-label like BraTS
                 output_pred_filename = f"{bundle_name}_multilabel_seg.nii.gz"

            output_pred_path = os.path.join(temp_bundle_output_dir, output_pred_filename)

            # Override dictionary for ConfigWorkflow
            # Input key ('image') and output key ('output_pred') are common but can vary per bundle.
            # CHECK THE BUNDLE'S metadata.json or inference/eval config for exact keys.
            override_dict = {
                "image": input_scan_path, # Common input key
                "output_pred": output_pred_path, # Common output key for prediction
                "device": self.device # Pass the determined device
            }
            
            # Cache or create ConfigWorkflow runner
            # Note: Caching ConfigWorkflow might be tricky if its internal state doesn't reset well
            # For simplicity in a hackathon, creating it each time might be safer unless performance is critical.
            # if bundle_name not in self.loaded_bundle_runners:
            runner = ConfigWorkflow(
                config_file=inference_config_path,
                workflow_type="infer", # or "eval" based on which config file was found
                **override_dict
            )
                # self.loaded_bundle_runners[bundle_name] = runner # Be cautious with caching runners
            # else:
            #     runner = self.loaded_bundle_runners[bundle_name]
            #     runner.update_config(override_dict) # Update paths for current run

            logger.info(f"Running MONAI bundle '{bundle_name}' on {input_scan_path}...")
            runner.run() # Execute the inference
            logger.info(f"MONAI bundle '{bundle_name}' finished. Output potentially at: {output_pred_path}")

            # Construct results
            # Output might be a single file path or more complex depending on the bundle
            if os.path.exists(output_pred_path):
                results['detections'] = [ # Representing segmentation as a "detection" of the structure
                    self.output_processor.format_detection_result(
                        class_name=str(task_config["output_structure_name"]), # Can be a list for BraTS
                        confidence=1.0, # Segmentation is a binary decision here
                        segmentation_info={"mask_file_temp": output_pred_path}
                    )
                ]
                results['raw_output_info'] = {
                    "output_pred_path_temp": output_pred_path,
                     "temp_output_dir_bundle": temp_bundle_output_dir # Return this for cleanup by caller
                }
            else:
                # Sometimes output path might be different or logged by the runner
                # Check runner.output_files or similar if available
                logger.warning(f"Expected output file {output_pred_path} not found for bundle {bundle_name}. Check bundle logs or config.")
                results['error'] = f"Output file not found at expected location: {output_pred_path}"
                results['raw_output_info'] = {
                    "expected_output_pred_path": output_pred_path,
                    "temp_output_dir_bundle": temp_bundle_output_dir
                }


        except Exception as e:
            logger.error(f"Error processing MONAI scan ({scan_type}) with bundle {bundle_name}: {e}", exc_info=True)
            results['error'] = f"MONAI processing failed: {str(e)}"
            # Ensure temp_bundle_output_dir is included for cleanup even on error if it was created
            if 'temp_bundle_output_dir' in locals():
                 results.setdefault('raw_output_info', {})['temp_output_dir_bundle'] = temp_bundle_output_dir

        # Metadata section
        processing_time = time.time() - start_time
        metadata = {
            "input_scan_path_provided": input_scan_path,
            "model_used": bundle_name,
            "model_type": "monai_bundle",
            "scan_type_processed": scan_type,
            "processing_time_seconds": round(processing_time, 3)
        }
        results['metadata'] = metadata
        
        # Caller (FastAPI endpoint) is responsible for managing temp_bundle_output_dir
        return results

    async def cleanup(self):
        logger.info("Cleaning up MONAI model handler...")
        self.loaded_bundle_runners.clear()
        self.model_configs.clear()
        if MONAI_AVAILABLE and self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("MONAI model handler cleaned up.")
