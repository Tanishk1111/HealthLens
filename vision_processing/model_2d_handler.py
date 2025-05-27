# model_2d_handler.py

import os
import io
import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

# Model imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("Ultralytics YOLO not available.")

try:
    import torch
    import torchvision.transforms as tv_transforms # Renamed to avoid conflict
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available.")

try:
    import torchxrayvision as xrv
    TORCHXRAYVISION_AVAILABLE = True
except ImportError:
    TORCHXRAYVISION_AVAILABLE = False
    logging.warning("TorchXRayVision not available.")

# --- Placeholder for your common utility classes ---
class ImageProcessor:
    def load_2d_image(self, image_bytes: bytes) -> np.ndarray:
        """Loads image bytes into a NumPy array (e.g., RGB or Grayscale)."""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            # Convert to a common format, e.g., L (grayscale) or RGB
            # For torchxrayvision, we'll handle specific conversion later,
            # but it's good to have a base format.
            if pil_image.mode not in ['L', 'RGB']:
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def extract_image_metadata(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extracts basic metadata."""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            return {
                "filename": filename,
                "format": pil_image.format,
                "mode": pil_image.mode,
                "width": pil_image.width,
                "height": pil_image.height,
                "size_bytes": len(image_bytes)
            }
        except Exception:
            return {"filename": filename, "error": "Could not extract metadata"}

class ModelOutputProcessor:
    def format_detection_result(self, class_name: str, confidence: float, 
                                bbox: Optional[List[float]], 
                                additional_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = {
            "class_name": class_name,
            "confidence": confidence,
            "bounding_box": bbox if bbox else "N/A", # Consistent key name
        }
        if additional_info:
            result.update(additional_info)
        return result

    def filter_detections_by_confidence(self, detections: List[Dict[str, Any]], 
                                        threshold: float) -> List[Dict[str, Any]]:
        return [d for d in detections if d.get('confidence', 0) >= threshold]

    def sort_detections_by_confidence(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)

    def calculate_detection_statistics(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not detections:
            return {"count": 0, "unique_classes": 0, "average_confidence": 0.0}
        
        class_counts = {}
        total_confidence = 0
        for det in detections:
            class_counts[det['class_name']] = class_counts.get(det['class_name'], 0) + 1
            total_confidence += det.get('confidence', 0)
        
        return {
            "total_detections_meeting_threshold": len(detections),
            "unique_classes_detected": len(class_counts),
            "average_confidence": round(total_confidence / len(detections), 4) if detections else 0.0,
            "class_distribution": class_counts
        }
# --- End of Placeholder classes ---


logger = logging.getLogger(__name__)
# Basic logging configuration for testing
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Model2DHandler:
    """Handler for 2D medical imaging models"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self.image_processor = ImageProcessor()
        self.output_processor = ModelOutputProcessor()
        self.is_initialized = False
        
        self.scan_type_to_model_key = {
            'xray_chest_2d': 'torchxrayvision_chest_all', # Default to the 'all' dataset model
            # Add other mappings as you implement more models
        }
        
        # Default class names (less relevant for XRV as it has its own)
        self.default_class_names = {
            # ... your other default class names ...
        }
    
    async def initialize(self):
        """Initialize 2D models"""
        if self.is_initialized:
            logger.info("2D model handler already initialized.")
            return

        logger.info("Initializing 2D model handler...")
        
        try:
            await self._load_torchxrayvision_models()
            # await self._load_yolo_models() # Call this when you are ready
            # await self._load_custom_models() # Call this when you are ready
            
            self.is_initialized = True
            logger.info(f"2D model handler initialized with {len(self.models)} model(s).")
            
        except Exception as e:
            logger.error(f"Error initializing 2D model handler: {e}", exc_info=True)
            # self.is_initialized = False # Ensure it's false on failure
            raise # Re-raise after logging for FastAPI to catch
    
    async def _load_torchxrayvision_models(self):
        """Load TorchXRayVision models for chest X-ray analysis"""
        if not TORCHXRAYVISION_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("TorchXRayVision or PyTorch not available, skipping XRV model loading.")
            return
            
        model_key = 'torchxrayvision_chest_all'
        try:
            # Load DenseNet model for chest X-ray pathology detection
            # This model is pre-trained on multiple datasets
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
            # Set model to evaluation mode immediately after loading
            model.eval() 
            self.models[model_key] = model
            
            # Configure model
            self.model_configs[model_key] = {
                'type': 'torchxrayvision',
                'class_names': {i: pathology for i, pathology in enumerate(model.pathologies)},
                'input_size': (224, 224), # Standard for this model
                'pathologies': model.pathologies, # Keep the list of pathology names
                'expected_input_channels': 1 # XRV models typically expect single channel
            }
            
            logger.info(f"Loaded TorchXRayVision DenseNet model ('{model_key}') with {len(model.pathologies)} pathologies.")
            
        except Exception as e:
            logger.error(f"Error loading TorchXRayVision model '{model_key}': {e}", exc_info=True)

    # --- Placeholder for _load_yolo_models and _load_custom_models ---
    async def _load_yolo_models(self):
        if not YOLO_AVAILABLE:
            logger.warning("YOLO (Ultralytics) not available, skipping YOLO model loading.")
            return
        logger.info("YOLO model loading (placeholder - not implemented yet).")
        # Implement your YOLO loading logic here, similar to your original file.
        # Ensure class names are correctly mapped.
        pass

    async def _load_custom_models(self):
        if not TORCH_AVAILABLE: # Or other relevant library for custom models
            logger.warning("PyTorch not available, skipping custom model loading.")
            return
        logger.info("Custom model loading (placeholder - not implemented yet).")
        # Implement your custom model loading here.
        pass
    # --- End of placeholders ---

    def is_ready(self) -> bool:
        """Check if handler is ready"""
        return self.is_initialized and bool(self.models) # Check if any models were actually loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        # ... (your existing get_model_info method is fine) ...
        info = {}
        for model_name, config in self.model_configs.items():
            info[model_name] = {
                'type': config.get('type', 'unknown'),
                'class_count': len(config.get('class_names', {})),
                'class_names': list(config.get('class_names', {}).values()), # List of names
                'input_size': config.get('input_size', 'unknown')
            }
        return info
    
    async def process_scan(
        self, 
        image_bytes: bytes, 
        scan_type: str, # e.g., "xray_chest_2d"
        filename_for_metadata: str = "uploaded_image.tmp",
        confidence_threshold: float = 0.1 # XRV models often have many low-confidence outputs
    ) -> Dict[str, Any]:
        """Process a 2D medical scan"""
        start_time = time.time()
        
        if not self.is_ready():
            logger.error("2D Model Handler is not initialized or has no models loaded.")
            raise RuntimeError("2D Model Handler not ready.") # Use RuntimeError for unrecoverable state

        try:
            model_key = self.scan_type_to_model_key.get(scan_type)
            if not model_key or model_key not in self.models:
                logger.warning(f"No specific model configured for scan_type '{scan_type}'.")
                # Fallback logic (optional, or raise error)
                # For now, let's try to use the first available model if one exists.
                if not self.models:
                    raise ValueError("No 2D models available for processing.")
                model_key = list(self.models.keys())[0]
                logger.warning(f"Falling back to using model: '{model_key}' for '{scan_type}'.")
            
            model = self.models[model_key]
            config = self.model_configs[model_key]
            
            # Load image (initial load to numpy array)
            # The specific model's inference function will handle detailed preprocessing
            image_pil = Image.open(io.BytesIO(image_bytes)) # Keep as PIL image for XRV
            
            detections: List[Dict[str, Any]] = []
            if config['type'] == 'torchxrayvision':
                detections = await self._run_torchxrayvision_inference(
                    model, image_pil, config # Pass PIL image
                )
            elif config['type'] == 'yolo':
                # Convert PIL to NumPy for consistency if your YOLO handler expects NumPy
                image_array = np.array(image_pil)
                detections = await self._run_yolo_inference(
                    model, image_array, config, confidence_threshold # Assuming YOLO takes np.ndarray
                )
            # Add elif for 'custom' type here
            else:
                logger.error(f"Unsupported model type '{config['type']}' for model '{model_key}'.")
                raise ValueError(f"Unsupported model type: {config['type']}")
            
            filtered_detections = self.output_processor.filter_detections_by_confidence(
                detections, confidence_threshold
            )
            sorted_detections = self.output_processor.sort_detections_by_confidence(
                filtered_detections
            )
            stats = self.output_processor.calculate_detection_statistics(sorted_detections)
            metadata = self.image_processor.extract_image_metadata(image_bytes, filename_for_metadata)
            metadata.update({
                'model_used': model_key,
                'model_type': config['type'],
                'scan_type_requested': scan_type, # The original requested scan type
                'confidence_threshold_applied': confidence_threshold
            })
            
            processing_time = time.time() - start_time
            
            return {
                'detections': sorted_detections,
                'statistics': stats,
                'metadata': metadata,
                'processing_time_seconds': round(processing_time, 3)
            }
            
        except ValueError as ve: # Catch specific, expected errors
            logger.error(f"ValueError processing 2D scan ({scan_type}): {ve}", exc_info=True)
            raise # Re-raise to be handled by FastAPI
        except Exception as e: # Catch unexpected errors
            logger.error(f"Unexpected error processing 2D scan ({scan_type}): {e}", exc_info=True)
            raise ValueError(f"Failed to process 2D scan due to an unexpected error: {str(e)}")

    # --- Placeholder for _run_yolo_inference and _run_custom_inference ---
    async def _run_yolo_inference(
        self, model, image: np.ndarray, config: Dict[str, Any], confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        logger.info("Running YOLO inference (placeholder - needs implementation from your original file).")
        # Implement your YOLO inference logic here using the `model` (Ultralytics YOLO object),
        # `image` (np.ndarray), `config` (for class_names), and `confidence_threshold`.
        # Ensure it returns a List[Dict[str, Any]] in the format expected by output_processor.
        return [] # Placeholder
    
    async def _run_custom_inference(
        self, model, image: np.ndarray, config: Dict[str, Any], confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        logger.info("Running custom model inference (placeholder - needs implementation).")
        return [] # Placeholder
    # --- End of placeholders ---

    async def _run_torchxrayvision_inference(
        self, 
        model: Any, # torch.nn.Module
        image_pil: Image.Image, # Expecting a PIL Image
        config: Dict[str, Any]
        # confidence_threshold is applied *after* getting all probabilities
    ) -> List[Dict[str, Any]]:
        """Run inference with TorchXRayVision model."""
        if not TORCH_AVAILABLE or not TORCHXRAYVISION_AVAILABLE:
            logger.error("Torch or TorchXRayVision not available for inference.")
            return []

        try:
            # Get pathologies from config
            pathologies = config.get('pathologies')
            if not pathologies:
                logger.error("Pathologies not found in model config for TorchXRayVision.")
                return []

            # === Correct Preprocessing for TorchXRayVision ===
            # 1. Convert to Grayscale if not already (XRV models typically expect 1 channel)
            if image_pil.mode != 'L':
                img_prepared = image_pil.convert('L')
            else:
                img_prepared = image_pil

            # 2. Use TorchXRayVision's recommended transforms
            # The 'DenseNet(weights="densenet121-res224-all")' model was trained with specific normalization
            # and resizing. TorchXRayVision handles this internally if you feed it correctly.
            # It expects a NumPy array [H, W] or [H, W, C] or PIL image.
            # We need to ensure it's a PyTorch tensor of shape [1, 1, H, W] (batch, channel, height, width)
            # The internal xrv.models.Resizer and xrv.models.XRayCenterCrop are often part of the
            # dataset preprocessing used during training.
            # For direct inference, ensure image is single channel, normalized [-1024, 1024] if from DICOM,
            # or [0,1] if a standard image format, and resized to 224x224.

            # Simplest way that often works if PIL image is 'L' mode, [0-255] range:
            transform = tv_transforms.Compose([
                tv_transforms.Resize(256), # Slightly larger before center crop
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(), # This converts PIL [0-255] to Tensor [0-1]
                # xrv.models.NumpyNormalize(), # Normalizes to [-1024, 1024] - use if input is like raw DICOM
                # If model was trained with ImageNet stats (some XRV models are):
                # tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                # However, for "densenet121-res224-all", this is often not needed as it learns from X-ray specific stats.
                # The ToTensor() followed by model's internal handling is usually sufficient.
                # If the model expects 3 channels duplicated from grayscale:
                # tv_transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.ndim == 3 and x.shape[0] == 1 else x)
            ])
            
            img_tensor = transform(img_prepared) # Shape: [C, H, W], C=1 for grayscale
            
            # Add batch dimension: [1, C, H, W]
            img_tensor = img_tensor.unsqueeze(0)

            # --- Device Handling (Optional but good practice) ---
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            img_tensor = img_tensor.to(device)
            # --- End Device Handling ---

            model.eval() # Ensure model is in evaluation mode
            
            with torch.no_grad():
                outputs = model(img_tensor) # Raw logits
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0] # Get probabilities for the first (only) image in batch

            detections = []
            for i, pathology_name in enumerate(pathologies):
                prob = float(probabilities[i])
                # The confidence threshold will be applied later by process_scan
                detection = self.output_processor.format_detection_result(
                    class_name=pathology_name,
                    confidence=prob,
                    bbox=None, # TorchXRayVision classification models don't output bounding boxes
                    additional_info={
                        'model_type': 'torchxrayvision',
                        'pathology_index': i,
                        # 'description': f"{pathology_name} probability: {prob:.4f}" # Optional
                    }
                )
                detections.append(detection)
            
            # logger.info(f"TorchXRayVision inference: {len(detections)} pathology probabilities generated.")
            return detections
            
        except Exception as e:
            logger.error(f"Error in TorchXRayVision inference: {e}", exc_info=True)
            # Return empty list on error to prevent breaking the whole request if one model fails
            return []


    async def cleanup(self):
        logger.info("Cleaning up 2D model handler...")
        # ... (your existing cleanup method is fine) ...
        pass # Replace with your original cleanup

# --- Example usage (for testing this file directly) ---
async def main_test():
    handler = Model2DHandler()
    await handler.initialize()

    if not handler.is_ready():
        logger.error("Handler not ready after initialization. Exiting test.")
        return

    print("\nLoaded Model Info:")
    print(handler.get_model_info())

    # Create a dummy grayscale PNG image for testing
    try:
        dummy_image_pil = Image.new('L', (512, 512), color='gray')
        img_byte_arr = io.BytesIO()
        dummy_image_pil.save(img_byte_arr, format='PNG')
        dummy_image_bytes = img_byte_arr.getvalue()
        
        logger.info("\nProcessing dummy chest X-ray (PNG)...")
        results = await handler.process_scan(
            image_bytes=dummy_image_bytes,
            scan_type='xray_chest_2d',
            filename_for_metadata="dummy_xray.png",
            confidence_threshold=0.05 # Lower threshold to see more raw outputs
        )
        
        print("\n--- Scan Results ---")
        print(f"Processing Time: {results.get('processing_time_seconds')}s")
        print(f"Model Used: {results.get('metadata', {}).get('model_used')}")
        print("Detections:")
        for det in results.get('detections', []):
            print(f"  - {det['class_name']}: {det['confidence']:.4f}")
        print("Statistics:")
        print(results.get('statistics'))

    except Exception as e:
        logger.error(f"Error during test processing: {e}", exc_info=True)
    finally:
        await handler.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test())
