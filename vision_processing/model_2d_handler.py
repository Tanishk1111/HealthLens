# vision_processing/model_2d_handler.py
import os
import io
import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

from .common import BaseImageProcessor, BaseModelOutputProcessor # Assuming common.py is in the same directory

# Model imports
try:
    import torch
    import torchvision.transforms as tv_transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch (torch or torchvision) not available. Some 2D models may not work.")

try:
    import torchxrayvision as xrv
    TORCHXRAYVISION_AVAILABLE = True
except ImportError:
    TORCHXRAYVISION_AVAILABLE = False
    logging.warning("TorchXRayVision not available. Chest X-ray models will be unavailable.")

logger = logging.getLogger(__name__)

class Model2DHandler:
    """Handler for 2D medical imaging models, primarily TorchXRayVision."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self.image_processor = BaseImageProcessor() # Use from common
        self.output_processor = BaseModelOutputProcessor() # Use from common
        self.is_initialized = False
        self.device = None # Will be set in initialize
        
        self.scan_type_to_model_key = {
            'xray_chest_pathology_2d': 'torchxrayvision_chest_all',
            # Add other 2D scan_type mappings here if you add more 2D models
        }
            
    async def initialize(self):
        if self.is_initialized:
            logger.info("2D model handler already initialized.")
            return

        logger.info("Initializing 2D model handler...")
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device} for 2D PyTorch models.")
        else:
            self.device = "cpu" # Fallback, though models won't load
            logger.warning("PyTorch not available, device set to CPU, PyTorch models will fail to load.")

        try:
            await self._load_torchxrayvision_models()
            # Add calls to load other 2D models here if needed
            
            self.is_initialized = True
            logger.info(f"2D model handler initialized with {len(self.models)} model(s).")
            
        except Exception as e:
            logger.error(f"Error initializing 2D model handler: {e}", exc_info=True)
            self.is_initialized = False # Ensure it's false on failure
            raise
    
    async def _load_torchxrayvision_models(self):
        if not TORCHXRAYVISION_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("TorchXRayVision or PyTorch not available, skipping XRV model loading.")
            return
            
        model_key = 'torchxrayvision_chest_all'
        try:
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            model.to(self.device)
            model.eval() 
            self.models[model_key] = model
            
            self.model_configs[model_key] = {
                'type': 'torchxrayvision_classification',
                'pathologies': model.pathologies,
                'input_size': (224, 224),
                'expected_input_channels': 1 
            }
            logger.info(f"Loaded TorchXRayVision model '{model_key}' to {self.device}.")
        except Exception as e:
            logger.error(f"Error loading TorchXRayVision model '{model_key}': {e}", exc_info=True)

    def is_ready(self) -> bool:
        return self.is_initialized and bool(self.models)
    
    def get_model_info(self) -> Dict[str, Any]:
        info = {}
        for model_key, config in self.model_configs.items():
            info[model_key] = {
                'type': config.get('type'),
                'pathologies_count': len(config.get('pathologies', [])),
                'pathologies_list': config.get('pathologies', []),
                'input_size': config.get('input_size')
            }
        return info
        
    async def process_scan(
        self, 
        image_bytes: bytes, 
        scan_type: str,
        filename_for_metadata: str = "uploaded_2d_image.tmp",
        confidence_threshold: float = 0.1 
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        if not self.is_ready():
            logger.error("2D Model Handler not ready.")
            raise RuntimeError("2D Model Handler not ready or no models loaded.")

        model_key = self.scan_type_to_model_key.get(scan_type)
        if not model_key or model_key not in self.models:
            logger.error(f"No 2D model configured for scan_type '{scan_type}'.")
            raise ValueError(f"Unsupported 2D scan_type: {scan_type}")
            
        model = self.models[model_key]
        config = self.model_configs[model_key]
        
        pil_image = self.image_processor.load_image_to_pil(image_bytes)
        if not pil_image:
            raise ValueError("Failed to load image for 2D processing.")

        detections: List[Dict[str, Any]] = []
        try:
            if config['type'] == 'torchxrayvision_classification':
                detections = await self._run_torchxrayvision_inference(model, pil_image, config)
            else:
                raise ValueError(f"Unsupported model type '{config['type']}' in 2D handler.")
        except Exception as e:
            logger.error(f"Error during 2D model inference for {scan_type}: {e}", exc_info=True)
            return {"error": f"Inference failed: {str(e)}"} # Return error structure

        filtered_detections = self.output_processor.filter_detections_by_confidence(
            detections, confidence_threshold
        )
        sorted_detections = self.output_processor.sort_detections_by_confidence(
            filtered_detections
        )
        
        processing_time = time.time() - start_time
        metadata = self.image_processor.extract_image_metadata(image_bytes, filename_for_metadata)
        metadata.update({
            'model_used': model_key,
            'model_type': config['type'],
            'scan_type_processed': scan_type,
            'confidence_threshold_applied': confidence_threshold,
            'processing_time_seconds': round(processing_time, 3)
        })
            
        return {
            'detections': sorted_detections, # Pathologies and their probabilities
            'metadata': metadata
        }

    async def _run_torchxrayvision_inference(
        self, model: Any, image_pil: Image.Image, config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not TORCH_AVAILABLE or not TORCHXRAYVISION_AVAILABLE:
            raise EnvironmentError("Torch or TorchXRayVision not available for inference.")

        pathologies = config.get('pathologies')
        if not pathologies:
            raise ValueError("Pathologies list not found in model config for TorchXRayVision.")

        # Preprocessing specific to torchxrayvision DenseNet models
        img_prepared = image_pil
        if img_prepared.mode != 'L': # Ensure grayscale
            img_prepared = img_prepared.convert('L')

        # Standard transforms for these models
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(), # Converts PIL [0-255] to Tensor [0-1.0]
            # For "densenet121-res224-all", specific normalization beyond ToTensor() is
            # often handled implicitly or a generic normalization can be used.
            # If issues arise, consider xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)
            # or check model-specific preprocessing if documented by torchxrayvision for that weight set.
            # tv_transforms.Normalize(mean=[0.5021], std=[0.2897]) # Example for NIH dataset mean/std if needed
        ])
        
        img_tensor = transform(img_prepared).unsqueeze(0).to(self.device) # Add batch dim, move to device

        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

        results = []
        for i, pathology_name in enumerate(pathologies):
            prob = float(probabilities[i])
            results.append(self.output_processor.format_detection_result(
                class_name=pathology_name,
                confidence=prob,
                # No bounding box or segmentation from these classification models
            ))
        return results

    async def cleanup(self):
        logger.info("Cleaning up 2D model handler...")
        self.models.clear()
        self.model_configs.clear()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("2D model handler cleaned up.")
