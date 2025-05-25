"""
3D Model Handler for HealthLens medical scan analysis.

Handles loading and inference for 3D medical imaging models including
MedYOLO, MONAI-based models, and other 3D medical AI models.
"""

import os
import io
import time
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Medical imaging imports
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    logging.warning("Nibabel not available")

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    logging.warning("PyDICOM not available")

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False
    logging.warning("SimpleITK not available")

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

try:
    import monai
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Spacing, 
        Orientation, ScaleIntensity, Resize
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logging.warning("MONAI not available")

from .common import ModelOutputProcessor

logger = logging.getLogger(__name__)

class Model3DHandler:
    """Handler for 3D medical imaging models"""
    
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.output_processor = ModelOutputProcessor()
        self.is_initialized = False
        
        # Model configurations for different scan types
        self.scan_type_to_model = {
            'ct_chest_3d': 'ct_chest_detector',
            'ct_abdomen_3d': 'ct_abdomen_detector',
            'ct_brain_3d': 'ct_brain_detector',
            'ct_spine_3d': 'ct_spine_detector',
            'mri_brain_3d': 'mri_brain_detector',
            'mri_spine_3d': 'mri_spine_detector',
            'mri_cardiac_3d': 'mri_cardiac_detector',
            'mri_abdomen_3d': 'mri_abdomen_detector',
            'pet_ct_3d': 'pet_ct_detector',
            'spect_3d': 'spect_detector'
        }
        
        # Default class names for different model types
        self.default_class_names = {
            'ct_chest_detector': {
                0: 'Normal_Lung',
                1: 'Nodule',
                2: 'Mass',
                3: 'Pneumonia',
                4: 'Pneumothorax',
                5: 'Pleural_Effusion',
                6: 'Atelectasis',
                7: 'Consolidation'
            },
            'ct_abdomen_detector': {
                0: 'Liver',
                1: 'Kidney',
                2: 'Spleen',
                3: 'Pancreas',
                4: 'Gallbladder',
                5: 'Liver_Lesion',
                6: 'Kidney_Lesion',
                7: 'Abdominal_Mass'
            },
            'ct_brain_detector': {
                0: 'Normal_Brain',
                1: 'Hemorrhage',
                2: 'Infarct',
                3: 'Tumor',
                4: 'Edema',
                5: 'Midline_Shift'
            },
            'mri_brain_detector': {
                0: 'Normal_Brain',
                1: 'Tumor',
                2: 'Lesion',
                3: 'Edema',
                4: 'Hemorrhage',
                5: 'Infarct'
            }
        }
        
        # Standard preprocessing parameters
        self.preprocessing_params = {
            'target_spacing': (1.0, 1.0, 1.0),  # mm
            'target_size': (128, 128, 128),     # voxels
            'intensity_range': (-1000, 1000),   # HU for CT
            'orientation': 'RAS'                 # Standard orientation
        }
    
    async def initialize(self):
        """Initialize 3D models"""
        logger.info("Initializing 3D model handler...")
        
        try:
            # Load available models
            await self._load_monai_models()
            await self._load_custom_3d_models()
            
            self.is_initialized = True
            logger.info(f"3D model handler initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error initializing 3D model handler: {e}")
            raise
    
    async def _load_monai_models(self):
        """Load MONAI-based 3D models"""
        if not MONAI_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("MONAI or PyTorch not available, skipping MONAI model loading")
            return
        
        model_paths = {
            'ct_chest_detector': os.getenv('CT_CHEST_MODEL_PATH', 'models/ct_chest_3d.pth'),
            'ct_abdomen_detector': os.getenv('CT_ABDOMEN_MODEL_PATH', 'models/ct_abdomen_3d.pth'),
            'mri_brain_detector': os.getenv('MRI_BRAIN_MODEL_PATH', 'models/mri_brain_3d.pth')
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    # Load MONAI model (placeholder - implement based on your model architecture)
                    # model = torch.load(model_path, map_location='cpu')
                    # self.models[model_name] = model
                    
                    self.model_configs[model_name] = {
                        'type': 'monai',
                        'class_names': self.default_class_names.get(model_name, {}),
                        'input_size': self.preprocessing_params['target_size'],
                        'spacing': self.preprocessing_params['target_spacing']
                    }
                    
                    logger.info(f"MONAI model path found for {model_name}, but loading not implemented")
                else:
                    logger.info(f"MONAI model not found: {model_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load MONAI model {model_name}: {e}")
    
    async def _load_custom_3d_models(self):
        """Load custom 3D models (e.g., MedYOLO-style models)"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping custom 3D model loading")
            return
        
        # Create placeholder models for demonstration
        for model_name in ['ct_chest_detector', 'ct_abdomen_detector']:
            if model_name not in self.models:
                # Create a placeholder model configuration
                self.model_configs[model_name] = {
                    'type': 'placeholder_3d',
                    'class_names': self.default_class_names.get(model_name, {}),
                    'input_size': self.preprocessing_params['target_size'],
                    'spacing': self.preprocessing_params['target_spacing']
                }
                
                # Add placeholder model (for demonstration)
                self.models[model_name] = "placeholder_model"
                logger.info(f"Created placeholder 3D model: {model_name}")
    
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
                'input_size': config.get('input_size', 'unknown'),
                'spacing': config.get('spacing', 'unknown')
            }
        return info
    
    async def process_scan(
        self, 
        file_bytes: bytes, 
        filename: str, 
        scan_type: str, 
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Process a 3D medical scan"""
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
                    raise ValueError("No models available for 3D processing")
            
            model = self.models[model_name]
            config = self.model_configs[model_name]
            
            # Load and preprocess 3D volume
            volume_data, metadata = await self._load_3d_volume(file_bytes, filename)
            preprocessed_volume = await self._preprocess_3d_volume(volume_data, config)
            
            # Run inference
            if config['type'] == 'monai':
                detections = await self._run_monai_inference(
                    model, preprocessed_volume, config, confidence_threshold
                )
            else:
                detections = await self._run_placeholder_3d_inference(
                    model, preprocessed_volume, config, confidence_threshold
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
            
            # Update metadata
            metadata.update({
                'model_used': model_name,
                'model_type': config['type'],
                'scan_type': scan_type,
                'preprocessing_applied': True
            })
            
            processing_time = time.time() - start_time
            
            return {
                'detections': sorted_detections,
                'statistics': stats,
                'metadata': metadata,
                'processing_time': round(processing_time, 3)
            }
            
        except Exception as e:
            logger.error(f"Error processing 3D scan: {e}")
            raise ValueError(f"Failed to process 3D scan: {e}")
    
    async def _load_3d_volume(self, file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load 3D volume from various formats"""
        filename_lower = filename.lower()
        
        try:
            # Create temporary file for libraries that need file paths
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                if filename_lower.endswith(('.nii', '.nii.gz')):
                    return await self._load_nifti(temp_path, filename)
                elif filename_lower.endswith(('.dcm', '.dicom')):
                    return await self._load_dicom(temp_path, filename)
                else:
                    raise ValueError(f"Unsupported 3D file format: {filename}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error loading 3D volume: {e}")
            raise ValueError(f"Failed to load 3D volume: {e}")
    
    async def _load_nifti(self, file_path: str, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load NIfTI file"""
        if not NIBABEL_AVAILABLE:
            raise ValueError("Nibabel not available for NIfTI loading")
        
        try:
            nifti_img = nib.load(file_path)
            volume_data = nifti_img.get_fdata()
            
            metadata = {
                'filename': filename,
                'format': 'NIfTI',
                'shape': volume_data.shape,
                'affine': nifti_img.affine.tolist(),
                'header': dict(nifti_img.header),
                'file_size_bytes': os.path.getsize(file_path)
            }
            
            logger.info(f"Loaded NIfTI volume with shape: {volume_data.shape}")
            return volume_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading NIfTI file: {e}")
            raise
    
    async def _load_dicom(self, file_path: str, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DICOM file"""
        if not PYDICOM_AVAILABLE:
            raise ValueError("PyDICOM not available for DICOM loading")
        
        try:
            # For single DICOM file (in real implementation, you'd handle DICOM series)
            dicom_data = pydicom.dcmread(file_path)
            
            if hasattr(dicom_data, 'pixel_array'):
                volume_data = dicom_data.pixel_array
                
                # Add third dimension if 2D DICOM
                if len(volume_data.shape) == 2:
                    volume_data = volume_data[np.newaxis, ...]
                
                metadata = {
                    'filename': filename,
                    'format': 'DICOM',
                    'shape': volume_data.shape,
                    'patient_id': getattr(dicom_data, 'PatientID', 'Unknown'),
                    'study_date': getattr(dicom_data, 'StudyDate', 'Unknown'),
                    'modality': getattr(dicom_data, 'Modality', 'Unknown'),
                    'file_size_bytes': os.path.getsize(file_path)
                }
                
                logger.info(f"Loaded DICOM volume with shape: {volume_data.shape}")
                return volume_data, metadata
            else:
                raise ValueError("DICOM file does not contain pixel data")
                
        except Exception as e:
            logger.error(f"Error loading DICOM file: {e}")
            raise
    
    async def _preprocess_3d_volume(self, volume: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        """Preprocess 3D volume for model inference"""
        try:
            # Ensure float32
            volume = volume.astype(np.float32)
            
            # Resize to target size
            target_size = config.get('input_size', self.preprocessing_params['target_size'])
            if volume.shape != target_size:
                # Simple resize using numpy (in production, use proper medical image resampling)
                from scipy.ndimage import zoom
                zoom_factors = [t/s for t, s in zip(target_size, volume.shape)]
                volume = zoom(volume, zoom_factors, order=1)
            
            # Intensity normalization (basic)
            volume = np.clip(volume, -1000, 1000)  # Clip to typical CT range
            volume = (volume + 1000) / 2000.0      # Normalize to [0, 1]
            
            # Add batch and channel dimensions
            volume = volume[np.newaxis, np.newaxis, ...]  # Shape: (1, 1, D, H, W)
            
            logger.info(f"Preprocessed 3D volume to shape: {volume.shape}")
            return volume
            
        except Exception as e:
            logger.error(f"Error preprocessing 3D volume: {e}")
            raise
    
    async def _run_monai_inference(
        self, 
        model, 
        volume: np.ndarray, 
        config: Dict[str, Any], 
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run inference with MONAI model"""
        # Placeholder for MONAI model inference
        logger.info("Running MONAI model inference (placeholder)")
        
        # Simulated output
        detections = [
            self.output_processor.format_detection_result(
                class_name="Liver",
                confidence=0.92,
                bbox=[20, 30, 40, 80, 90, 100],  # 3D bbox: [xmin, ymin, zmin, xmax, ymax, zmax]
                additional_info={'model_type': 'monai', 'volume_region': 'abdomen'}
            ),
            self.output_processor.format_detection_result(
                class_name="Kidney",
                confidence=0.87,
                bbox=[60, 70, 45, 90, 100, 85],
                additional_info={'model_type': 'monai', 'volume_region': 'abdomen'}
            )
        ]
        
        return detections
    
    async def _run_placeholder_3d_inference(
        self, 
        model, 
        volume: np.ndarray, 
        config: Dict[str, Any], 
        confidence_threshold: float
    ) -> List[Dict[str, Any]]:
        """Run inference with placeholder 3D model"""
        logger.info("Running placeholder 3D model inference")
        
        class_names = config.get('class_names', {})
        
        # Simulate some detections based on volume analysis
        detections = []
        
        # Simulate finding structures in different regions
        for i, (class_id, class_name) in enumerate(class_names.items()):
            if i >= 3:  # Limit to 3 detections for demo
                break
                
            # Generate random but plausible bounding box
            z_center = 32 + i * 20
            y_center = 64 + i * 15
            x_center = 64 + i * 10
            
            bbox = [
                max(0, x_center - 15), max(0, y_center - 15), max(0, z_center - 10),
                min(127, x_center + 15), min(127, y_center + 15), min(127, z_center + 10)
            ]
            
            confidence = 0.8 - i * 0.1  # Decreasing confidence
            
            if confidence >= confidence_threshold:
                detection = self.output_processor.format_detection_result(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    additional_info={
                        'model_type': 'placeholder_3d',
                        'class_id': class_id,
                        'volume_shape': volume.shape
                    }
                )
                detections.append(detection)
        
        return detections
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up 3D model handler...")
        
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
        
        logger.info("3D model handler cleanup completed") 