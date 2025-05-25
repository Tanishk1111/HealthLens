"""
2D Model Handler for HealthLens medical scan analysis.

Handles loading and inference for 2D medical imaging models including
YOLO-based detectors, MedSAM, and other 2D medical AI models.
"""

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
    logging.warning("Ultralytics YOLO not available")

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

from .common import ImageProcessor, ModelOutputProcessor

logger = logging.getLogger(__name__)

class Model2DHandler:
    """Handler for 2D medical imaging models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.image_processor = ImageProcessor()
        self.output_processor = ModelOutputProcessor()
        self.is_initialized = False
        
        # Model configurations for different scan types
        self.scan_type_to_model = {
            'xray_chest_2d': 'chest_xray_detector',
            'xray_abdomen_2d': 'general_xray_detector',
            'xray_bone_2d': 'bone_xray_detector',
            'ultrasound_abdomen_2d': 'ultrasound_detector',
            'ultrasound_cardiac_2d': 'cardiac_ultrasound_detector',
            'ultrasound_obstetric_2d': 'obstetric_ultrasound_detector',
            'mammography_2d': 'mammography_detector',
            'fundus_2d': 'fundus_detector',
            'dermatology_2d': 'dermatology_detector',
            'endoscopy_2d': 'endoscopy_detector'
        }
        
        # Default class names for different model types
        self.default_class_names = {
            'chest_xray_detector': {
                0: 'Normal',
                1: 'Pneumonia',
                2: 'Pneumothorax',
                3: 'Pleural_Effusion',
                4: 'Cardiomegaly',
                5: 'Nodule',
                6: 'Mass',
                7: 'Atelectasis',
                8: 'Consolidation'
            },
            'general_xray_detector': {
                0: 'Normal',
                1: 'Abnormal_Finding',
                2: 'Fracture',
                3: 'Foreign_Object'
            },
            'ultrasound_detector': {
                0: 'Normal_Structure',
                1: 'Cyst',
                2: 'Mass',
                3: 'Calcification',
                4: 'Fluid_Collection'
            },
            'mammography_detector': {
                0: 'Normal',
                1: 'Mass',
                2: 'Calcification',
                3: 'Architectural_Distortion',
                4: 'Asymmetry'
            },
            'dermatology_detector': {
                0: 'Normal_Skin',
                1: 'Melanoma',
                2: 'Nevus',
                3: 'Basal_Cell_Carcinoma',
                4: 'Actinic_Keratosis',
                5: 'Seborrheic_Keratosis'
            }
        }
    
    async def initialize(self):
        """Initialize 2D models"""
        logger.info("Initializing 2D model handler...")
        
        try:
            # Load available models
            await self._load_yolo_models()
            await self._load_custom_models()
            
            self.is_initialized = True
            logger.info(f"2D model handler initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error initializing 2D model handler: {e}")
            raise
    
    async def _load_yolo_models(self):
        """Load YOLO-based medical models"""
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available, skipping YOLO model loading")
            return
        
        model_paths = {
            'chest_xray_detector': os.getenv('CHEST_XRAY_MODEL_PATH', 'models/chest_xray_yolo.pt'),
            'general_xray_detector': os.getenv('GENERAL_XRAY_MODEL_PATH', 'models/general_xray_yolo.pt'),
            'ultrasound_detector': os.getenv('ULTRASOUND_MODEL_PATH', 'models/ultrasound_yolo.pt')
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    model = YOLO(model_path)
                    self.models[model_name] = model
                    
                    # Get class names from model or use defaults
                    if hasattr(model, 'names') and model.names:
                        self.model_configs[model_name] = {
                            'type': 'yolo',
                            'class_names': model.names,
                            'input_size': (640, 640)  # Default YOLO input size
                        }
                    else:
                        self.model_configs[model_name] = {
                            'type': 'yolo',
                            'class_names': self.default_class_names.get(model_name, {}),
                            'input_size': (640, 640)
                        }
                    
                    logger.info(f"Loaded YOLO model: {model_name}")
                else:
                    # Use default YOLOv8 model as fallback
                    if model_name == 'chest_xray_detector':
                        model = YOLO('yolov8n.pt')  # Default model
                        self.models[model_name] = model
                        self.model_configs[model_name] = {
                            'type': 'yolo',
                            'class_names': self.default_class_names.get(model_name, model.names),
                            'input_size': (640, 640)
                        }
                        logger.info(f"Loaded default YOLO model for: {model_name}")
                        
            except Exception as e:
                logger.warning(f"Could not load YOLO model {model_name}: {e}")
    
    async def _load_custom_models(self):
        """Load custom PyTorch models"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping custom model loading")
            return
        
        # Add custom model loading logic here
        # This is where you would load MedSAM, custom CNN models, etc.
        
        custom_model_configs = {
            'mammography_detector': {
                'path': os.getenv('MAMMOGRAPHY_MODEL_PATH', 'models/mammography_model.pth'),
                'type': 'custom_cnn'
            },
            'dermatology_detector': {
                'path': os.getenv('DERMATOLOGY_MODEL_PATH', 'models/dermatology_model.pth'),
                'type': 'custom_cnn'
            }
        }
        
        for model_name, config in custom_model_configs.items():
            try:
                if os.path.exists(config['path']):
                    # Load custom model (placeholder - implement based on your model architecture)
                    # model = torch.load(config['path'])
                    # self.models[model_name] = model
                    # self.model_configs[model_name] = config
                    logger.info(f"Custom model path found for {model_name}, but loading not implemented")
                else:
                    logger.info(f"Custom model not found: {config['path']}")
                    
            except Exception as e:
                logger.warning(f"Could not load custom model {model_name}: {e}")
    
    def is_ready(self) -> bool:
        """Check if handler is ready"""
        return self.is_initialized and len(self.models) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {}
        for model_name, config in self.model_configs.items():
            info[model_name] = {
                'type': config.get('type', 'unknown'),
                'class_count': len(config.get('class_names', {})),
                'class_names': list(config.get('class_names', {}).values()),
                'input_size': config.get('input_size', 'unknown')
            }
        return info
    
    async def process_scan(
        self, 
        image_bytes: bytes, 
        scan_type: str, 
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Process a 2D medical scan"""
        start_time = time.time()
        
        try:
            # Get appropriate model for scan type
            model_name = self.scan_type_to_model.get(scan_type)
            if not model_name or model_name not in self.models:
                # Use first available model as fallback
                if self.models:
                    model_name = list(self.models.keys())[0]
                    logger.warning(f"No specific model for {scan_type}, using {model_name}")
                else:
                    raise ValueError("No models available for 2D processing")
            
            model = self.models[model_name]
            config = self.model_configs[model_name]
            
            # Load and preprocess image
            image_array = self.image_processor.load_2d_image(image_bytes)
            
            # Run inference based on model type
            if config['type'] == 'yolo':
                detections = await self._run_yolo_inference(
                    model, image_array, config, confidence_threshold
                )
            else:
                detections = await self._run_custom_inference(
                    model, image_array, config, confidence_threshold
                )
            
            # Process results
            filtered_detections = self.output_processor.filter_detections_by_confidence(
                detections, confidence_threshold
            )
            sorted_detections = self.output_processor.sort_detections_by_confidence(
                filtered_detections
            )
            
            # Calculate statistics
            stats = self.output_processor.calculate_detection_statistics(sorted_detections)
            
            # Extract metadata
            metadata = self.image_processor.extract_image_metadata(image_bytes, "uploaded_image")
            metadata.update({
                'model_used': model_name,
                'model_type': config['type'],
                'scan_type': scan_type
            })
            
            processing_time = time.time() - start_time
            
            return {
                'detections': sorted_detections,
                'statistics': stats,
                'metadata': metadata,
                'processing_time': round(processing_time, 3)
            }
            
        except Exception as e:
            logger.error(f"Error processing 2D scan: {e}")
            raise ValueError(f"Failed to process 2D scan: {e}")
    
    async def _run_yolo_inference(
        self, 
        model, 
        image: np.ndarray, 
        config: Dict[str, Any], 
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run inference with YOLO model"""
        try:
            # Convert numpy array to PIL Image for YOLO
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image)
            
            # Run inference
            results = model(pil_image, conf=confidence_threshold, verbose=False)
            
            detections = []
            class_names = config.get('class_names', {})
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        bbox = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                        
                        class_name = class_names.get(class_id, f"Class_{class_id}")
                        
                        detection = self.output_processor.format_detection_result(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                            additional_info={
                                'class_id': class_id,
                                'model_type': 'yolo'
                            }
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in YOLO inference: {e}")
            raise
    
    async def _run_custom_inference(
        self, 
        model, 
        image: np.ndarray, 
        config: Dict[str, Any], 
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run inference with custom model"""
        # Placeholder for custom model inference
        # Implement based on your specific model architecture
        
        logger.info("Running custom model inference (placeholder)")
        
        # Simulated output for demonstration
        detections = [
            self.output_processor.format_detection_result(
                class_name="Simulated_Finding",
                confidence=0.85,
                bbox=[100, 100, 200, 200],
                additional_info={'model_type': 'custom'}
            )
        ]
        
        return detections
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up 2D model handler...")
        
        # Clear models from memory
        for model_name in list(self.models.keys()):
            try:
                del self.models[model_name]
            except Exception as e:
                logger.warning(f"Error cleaning up model {model_name}: {e}")
        
        self.models.clear()
        self.model_configs.clear()
        self.is_initialized = False
        
        # Force garbage collection if torch is available
        if TORCH_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("2D model handler cleanup completed") 