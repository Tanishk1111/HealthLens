# vision_processing/common.py
import logging
import io
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class BaseImageProcessor:
    """Basic image processing utilities."""

    def load_image_to_pil(self, image_bytes: bytes) -> Optional[Image.Image]:
        """Loads image bytes into a PIL Image object."""
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            return pil_image
        except Exception as e:
            logger.error(f"Error loading image to PIL: {e}")
            return None

    def pil_to_numpy(self, pil_image: Image.Image, mode: str = 'RGB') -> Optional[np.ndarray]:
        """Converts PIL image to NumPy array, optionally converting mode."""
        try:
            if pil_image.mode != mode and mode is not None:
                pil_image = pil_image.convert(mode)
            return np.array(pil_image)
        except Exception as e:
            logger.error(f"Error converting PIL to NumPy: {e}")
            return None

    def extract_image_metadata(self, image_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extracts basic metadata from image bytes using PIL."""
        pil_image = self.load_image_to_pil(image_bytes)
        if pil_image:
            return {
                "filename": filename,
                "format": pil_image.format,
                "mode": pil_image.mode,
                "width": pil_image.width,
                "height": pil_image.height,
                "size_bytes": len(image_bytes)
            }
        return {"filename": filename, "error": "Could not extract metadata"}


class BaseModelOutputProcessor:
    """Processes and formats model outputs."""

    def format_detection_result(
        self, 
        class_name: str, 
        confidence: float, 
        bbox: Optional[List[float]] = None, 
        segmentation_info: Optional[Any] = None, # e.g., path to mask file
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = {
            "class_name": class_name,
            "confidence": round(confidence, 4), # Standardize confidence format
            "bounding_box": bbox if bbox else "N/A",
            "segmentation_info": segmentation_info if segmentation_info else "N/A"
        }
        if additional_info:
            result.update(additional_info)
        return result

    def filter_detections_by_confidence(
        self, detections: List[Dict[str, Any]], threshold: float
    ) -> List[Dict[str, Any]]:
        return [d for d in detections if d.get('confidence', 0.0) >= threshold]

    def sort_detections_by_confidence(
        self, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True)

    def summarize_output(self, vision_results: Dict[str, Any]) -> str:
        """Creates a simple textual summary of vision results for Sonar."""
        summary_parts = []
        
        model_used = vision_results.get("metadata", {}).get("model_used", "Unknown model")
        summary_parts.append(f"Analysis performed using {model_used}.")

        detections = vision_results.get("detections")
        if detections:
            summary_parts.append("Key findings include:")
            for det in detections[:3]: # Summarize top 3 for brevity
                name = det.get('class_name', 'Unknown finding')
                conf = det.get('confidence', 0.0)
                summary_parts.append(f"- {name} (confidence: {conf:.2f})")
            if len(detections) > 3:
                summary_parts.append(f"...and {len(detections) - 3} other findings.")
        
        segmented_structures = vision_results.get("segmented_structures")
        if segmented_structures:
            if isinstance(segmented_structures, list):
                 summary_parts.append(f"Segmented structures: {', '.join(segmented_structures)}.")
            else: # If it's a path or other info
                summary_parts.append(f"Segmentation output generated: {segmented_structures}.")
        
        if not detections and not segmented_structures and "error" not in vision_results:
            summary_parts.append("No specific anomalies or structures highlighted by the model based on current thresholds.")
        elif "error" in vision_results:
            summary_parts.append(f"An error occurred during vision processing: {vision_results['error']}")
            
        return " ".join(summary_parts)
