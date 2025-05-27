"""
Common utilities for vision processing in HealthLens.
"""

import io
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)

class ScanTypeValidator:
    """Validates and categorizes medical scan types"""
    
    def __init__(self):
        self.supported_2d_types = {
            'xray_chest_2d': 'Chest X-ray',
            'xray_abdomen_2d': 'Abdominal X-ray',
            'xray_bone_2d': 'Bone X-ray',
            'ultrasound_abdomen_2d': 'Abdominal Ultrasound',
            'ultrasound_cardiac_2d': 'Cardiac Ultrasound',
            'ultrasound_obstetric_2d': 'Obstetric Ultrasound',
            'mammography_2d': 'Mammography',
            'fundus_2d': 'Fundus Photography',
            'dermatology_2d': 'Dermatological Image',
            'endoscopy_2d': 'Endoscopic Image'
        }
        
        self.supported_3d_types = {
            'ct_chest_3d': 'Chest CT Scan',
            'ct_abdomen_3d': 'Abdominal CT Scan',
            'ct_brain_3d': 'Brain CT Scan',
            'ct_spine_3d': 'Spine CT Scan',
            'mri_brain_3d': 'Brain MRI',
            'mri_spine_3d': 'Spine MRI',
            'mri_cardiac_3d': 'Cardiac MRI',
            'mri_abdomen_3d': 'Abdominal MRI',
            'pet_ct_3d': 'PET-CT Scan',
            'spect_3d': 'SPECT Scan'
        }
    
    def is_valid_scan_type(self, scan_type: str) -> bool:
        """Check if scan type is supported"""
        return scan_type in self.supported_2d_types or scan_type in self.supported_3d_types
    
    def is_3d_scan_type(self, scan_type: str) -> bool:
        """Check if scan type is 3D"""
        return scan_type in self.supported_3d_types
    
    def get_supported_types(self) -> Dict[str, str]:
        """Get all supported scan types"""
        return {**self.supported_2d_types, **self.supported_3d_types}
    
    def get_2d_types(self) -> Dict[str, str]:
        """Get supported 2D scan types"""
        return self.supported_2d_types.copy()
    
    def get_3d_types(self) -> Dict[str, str]:
        """Get supported 3D scan types"""
        return self.supported_3d_types.copy()
    
    def get_scan_description(self, scan_type: str) -> str:
        """Get human-readable description of scan type"""
        all_types = self.get_supported_types()
        return all_types.get(scan_type, "Unknown scan type")

class ImageProcessor:
    """Common image processing utilities"""
    
    def __init__(self):
        self.supported_2d_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.supported_3d_formats = {'.nii', '.nii.gz', '.dcm', '.dicom'}
    
    def validate_file_format(self, filename: str, is_3d: bool = False) -> bool:
        """Validate if file format is supported"""
        filename_lower = filename.lower()
        
        if is_3d:
            return any(filename_lower.endswith(fmt) for fmt in self.supported_3d_formats)
        else:
            return any(filename_lower.endswith(fmt) for fmt in self.supported_2d_formats)
    
    def load_2d_image(self, image_bytes: bytes) -> np.ndarray:
        """Load 2D image from bytes"""
        try:
            # Load with PIL first
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(pil_image)
            
            logger.info(f"Loaded 2D image with shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading 2D image: {e}")
            raise ValueError(f"Failed to load 2D image: {e}")
    
    def preprocess_2d_image(self, image: np.ndarray, target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """Preprocess 2D image for model inference"""
        try:
            # Resize image
            if image.shape[:2] != target_size:
                image = cv2.resize(image, target_size)
            
            # Normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Ensure 3 channels
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing 2D image: {e}")
            raise ValueError(f"Failed to preprocess 2D image: {e}")
    
    def extract_image_metadata(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extract metadata from image"""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            metadata = {
                'filename': filename,
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size': pil_image.size,
                'file_size_bytes': len(image_bytes)
            }
            
            # Extract EXIF data if available
            if hasattr(pil_image, '_getexif') and pil_image._getexif():
                metadata['exif'] = dict(pil_image._getexif())
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
            return {
                'filename': filename,
                'file_size_bytes': len(image_bytes),
                'error': str(e)
            }

class ModelOutputProcessor:
    """Process and format model outputs"""
    
    @staticmethod
    def format_detection_result(
        class_name: str,
        confidence: float,
        bbox: Optional[List[float]],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format a single detection result"""
        result = {
            'class_name': class_name,
            'confidence': round(confidence, 4)
        }
        
        # Only add bounding box if provided (some models like TorchXRayVision don't provide boxes)
        if bbox is not None:
            result['bounding_box'] = [round(coord, 2) for coord in bbox]
        else:
            result['bounding_box'] = None
        
        if additional_info:
            result.update(additional_info)
        
        return result
    
    @staticmethod
    def filter_detections_by_confidence(
        detections: List[Dict[str, Any]], 
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Filter detections by confidence threshold"""
        return [det for det in detections if det.get('confidence', 0) >= threshold]
    
    @staticmethod
    def sort_detections_by_confidence(
        detections: List[Dict[str, Any]], 
        descending: bool = True
    ) -> List[Dict[str, Any]]:
        """Sort detections by confidence"""
        return sorted(
            detections, 
            key=lambda x: x.get('confidence', 0), 
            reverse=descending
        )
    
    @staticmethod
    def group_detections_by_class(
        detections: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group detections by class name"""
        grouped = {}
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(detection)
        return grouped
    
    @staticmethod
    def calculate_detection_statistics(
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistics for detections"""
        if not detections:
            return {
                'total_detections': 0,
                'unique_classes': 0,
                'avg_confidence': 0,
                'max_confidence': 0,
                'min_confidence': 0
            }
        
        confidences = [det.get('confidence', 0) for det in detections]
        unique_classes = set(det.get('class_name', 'unknown') for det in detections)
        
        return {
            'total_detections': len(detections),
            'unique_classes': len(unique_classes),
            'class_names': list(unique_classes),
            'avg_confidence': round(np.mean(confidences), 4),
            'max_confidence': round(max(confidences), 4),
            'min_confidence': round(min(confidences), 4)
        } 