# sonar_integration/client.py
import os
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SonarClient:
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.api_url = api_url or "https://api.perplexity.ai/chat/completions" # Default URL
        
        if not self.api_key:
            logger.error("Perplexity API key not found. Sonar client will not function.")
            # raise ValueError("Perplexity API key is required.") # Or handle gracefully

    def _generate_base_prompt(self, scan_type_description: str, findings_summary: str) -> str:
        return (
            f"A medical scan ({scan_type_description}) was analyzed. Key findings include: {findings_summary}. "
            "Based on these findings, what are the potential implications, relevant medical conditions, "
            "or areas of research? Please provide concise information and cite sources if possible. "
            "Consider this information is for a healthcare professional."
        )

    def _generate_patient_explanation_prompt(self, scan_type_description: str, findings_summary: str) -> str:
        return (
            f"Explain the following findings from a medical scan ({scan_type_description}) in simple, "
            f"easy-to-understand terms for a patient: {findings_summary}. Avoid overly technical jargon."
        )

    def prepare_sonar_query_from_vision_results(
        self, 
        vision_results: Dict[str, Any], 
        scan_type: str, # e.g., "xray_chest_pathology_2d"
        query_type: str = "professional" # "professional" or "patient"
    ) -> str:
        """
        Generates a natural language query for Sonar based on vision model detections.
        """
        if vision_results.get("error"):
            return f"The analysis of the {scan_type} encountered an error: {vision_results['error']}. What are common issues or troubleshooting steps for this type of scan analysis?"

        findings_summary_parts = []
        
        # Process detections (typically from 2D models like TorchXRayVision)
        detections = vision_results.get('detections', [])
        if detections:
            # Summarize top N high-confidence detections
            count = 0
            for det in sorted(detections, key=lambda x: x.get('confidence', 0.0), reverse=True):
                if det.get('confidence', 0.0) >= 0.1: # Basic threshold for including in summary
                    findings_summary_parts.append(f"{det.get('class_name', 'Unknown Finding')} (confidence: {det.get('confidence', 0.0):.2f})")
                    count += 1
                    if count >= 3: # Limit to top 3 for brevity
                        break
        
        # Process segmentations (typically from 3D models like TotalSegmentator or MONAI)
        raw_output_info = vision_results.get('raw_output_info', {})
        if 'segmented_structures_list' in raw_output_info: # From TotalSegmentator
            structures = raw_output_info['segmented_structures_list']
            if structures:
                findings_summary_parts.append(f"Key anatomical structures segmented: {', '.join(structures[:5])}" + ("..." if len(structures) > 5 else ""))
        
        elif 'output_pred_path_temp' in raw_output_info: # From MONAI
            # Find the original 'detections' entry for MONAI which contains class_name
            monai_detection_info = next((d for d in detections if "mask_file_temp" in d.get("segmentation_info", {})), None)
            if monai_detection_info:
                structure_name = monai_detection_info.get("class_name", "a structure")
                if isinstance(structure_name, list): # For multi-label like BraTS
                    findings_summary_parts.append(f"Segmentation of {', '.join(structure_name)} was performed.")
                else:
                    findings_summary_parts.append(f"Segmentation of {structure_name} was performed.")


        if not findings_summary_parts:
            findings_text = "No specific findings highlighted by the vision model at current thresholds, or analysis focused on general segmentation."
        else:
            findings_text = ", ".join(findings_summary_parts)

        if query_type == "patient":
            return self._generate_patient_explanation_prompt(scan_type, findings_text)
        else: # Default to professional
            return self._generate_base_prompt(scan_type, findings_text)

    async def query_sonar(self, query_text: str, model: str = "sonar-medium-online") -> Dict[str, Any]:
        if not self.api_key:
            logger.error("Cannot query Sonar: API key missing.")
            return {"error": "Sonar API key not configured."}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }
        # This payload structure is typical for Perplexity chat completions API
        payload = {
            "model": model, 
            "messages": [
                {"role": "system", "content": "You are an AI assistant providing information based on medical imaging analysis findings. Be concise, informative, and cite sources if applicable."},
                {"role": "user", "content": query_text},
            ],
            # "max_tokens": 500, # Optional
        }

        try:
            # Using httpx for async requests if this client is called from async FastAPI
            # For simplicity here, using synchronous 'requests'. If used in async FastAPI routes,
            # run this in a threadpool or use an async HTTP client like 'httpx'.
            # For now, this will block the event loop if called directly from an async function.
            # Consider: loop = asyncio.get_event_loop(); response = await loop.run_in_executor(None, requests.post, self.api_url, headers, json_payload)
            
            logger.info(f"Sending query to Sonar ({model}): {query_text[:100]}...") # Log snippet
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=45)
            response.raise_for_status()  # Raise an exception for HTTP errors 4xx/5xx
            
            response_data = response.json()
            # Extract the main content - structure may vary slightly based on Perplexity API version
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                message_content = response_data["choices"][0].get("message", {}).get("content", "")
                return {"explanation": message_content, "raw_response": response_data}
            else:
                logger.warning(f"Sonar response structure unexpected: {response_data}")
                return {"error": "Unexpected response structure from Sonar.", "raw_response": response_data}

        except requests.exceptions.Timeout:
            logger.error("Perplexity API request timed out.")
            return {"error": "Sonar API request timed out."}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Perplexity API: {e}", exc_info=True)
            error_details = str(e)
            if e.response is not None:
                try:
                    error_details = e.response.json()
                except ValueError:
                    error_details = e.response.text
            return {"error": "Failed to get response from Sonar API.", "details": error_details}
        except Exception as e:
            logger.error(f"An unexpected error occurred in Sonar client: {e}", exc_info=True)
            return {"error": f"Unexpected Sonar client error: {str(e)}"}
