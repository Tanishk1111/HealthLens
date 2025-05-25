"""
Perplexity Sonar API client for HealthLens medical consultation.

Handles communication with Perplexity's Sonar API to provide expert-level
medical insights and research context for detected findings.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
import requests
import aiohttp
import json

logger = logging.getLogger(__name__)

class PerplexityClient:
    """Client for Perplexity Sonar API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.session = None
        
        # Model configurations
        self.models = {
            'sonar_medium': 'sonar-medium-online',
            'sonar_large': 'sonar-large-online',
            'sonar_huge': 'sonar-huge-online'
        }
        
        # Default model for different types of queries
        self.default_model = self.models['sonar_medium']
        
        # Request configuration
        self.default_config = {
            'temperature': 0.3,  # Lower temperature for more factual responses
            'max_tokens': 1500,
            'top_p': 0.9
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def _generate_medical_query(
        self, 
        detections: List[Dict[str, Any]], 
        scan_type: str, 
        dimensionality: str
    ) -> str:
        """Generate a medical query for Sonar based on detections"""
        
        if not detections:
            return (
                f"I have analyzed a {dimensionality} {scan_type} medical scan but no specific "
                f"findings were detected above the confidence threshold. What are the common "
                f"normal variants, artifacts, and key areas to evaluate in {scan_type} imaging? "
                f"Please provide clinical context and interpretation guidelines."
            )
        
        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Group by class name and get top findings
        findings_summary = {}
        for detection in sorted_detections[:5]:  # Top 5 findings
            class_name = detection.get('class_name', 'Unknown')
            confidence = detection.get('confidence', 0)
            
            if class_name not in findings_summary:
                findings_summary[class_name] = {
                    'count': 0,
                    'max_confidence': 0,
                    'avg_confidence': 0,
                    'confidences': []
                }
            
            findings_summary[class_name]['count'] += 1
            findings_summary[class_name]['confidences'].append(confidence)
            findings_summary[class_name]['max_confidence'] = max(
                findings_summary[class_name]['max_confidence'], confidence
            )
        
        # Calculate averages
        for finding in findings_summary.values():
            finding['avg_confidence'] = sum(finding['confidences']) / len(finding['confidences'])
        
        # Build query
        findings_text = []
        for class_name, stats in findings_summary.items():
            if stats['count'] == 1:
                findings_text.append(
                    f"{class_name} (confidence: {stats['max_confidence']:.2f})"
                )
            else:
                findings_text.append(
                    f"{class_name} ({stats['count']} instances, "
                    f"max confidence: {stats['max_confidence']:.2f}, "
                    f"avg confidence: {stats['avg_confidence']:.2f})"
                )
        
        findings_str = "; ".join(findings_text)
        
        query = (
            f"I have analyzed a {dimensionality} {scan_type} medical scan using AI detection models. "
            f"The following findings were identified: {findings_str}. "
            f"\n\nPlease provide:\n"
            f"1. Clinical significance and differential diagnosis for these findings\n"
            f"2. Recommended follow-up or additional imaging if needed\n"
            f"3. Key clinical correlations to consider\n"
            f"4. Any limitations or considerations when interpreting AI-detected findings\n"
            f"5. Recent research or guidelines relevant to these findings\n\n"
            f"Please cite medical literature and provide evidence-based recommendations "
            f"suitable for both radiologists and referring physicians."
        )
        
        return query
    
    async def analyze_findings(
        self, 
        detections: List[Dict[str, Any]], 
        scan_type: str, 
        dimensionality: str,
        model: Optional[str] = None,
        custom_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze medical findings using Perplexity Sonar
        
        Args:
            detections: List of detected findings from vision models
            scan_type: Type of medical scan
            dimensionality: "2D" or "3D"
            model: Specific Sonar model to use
            custom_query: Custom query instead of auto-generated one
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Generate or use custom query
            if custom_query:
                query = custom_query
            else:
                query = self._generate_medical_query(detections, scan_type, dimensionality)
            
            # Select model
            selected_model = model or self.default_model
            
            # Prepare request
            payload = {
                'model': selected_model,
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            "You are an expert medical AI assistant specializing in medical imaging "
                            "interpretation and clinical decision support. Provide evidence-based, "
                            "accurate medical information with appropriate citations. Always emphasize "
                            "that AI findings should be correlated with clinical context and that "
                            "final interpretation should be made by qualified medical professionals. "
                            "Structure your responses clearly with clinical significance, differential "
                            "diagnosis, and recommendations."
                        )
                    },
                    {
                        'role': 'user',
                        'content': query
                    }
                ],
                **self.default_config
            }
            
            # Make request
            if self.session:
                # Use async session if available
                response_data = await self._make_async_request(payload)
            else:
                # Fall back to synchronous request
                response_data = await self._make_sync_request(payload)
            
            # Process response
            return self._process_response(response_data, query, selected_model, detections)
            
        except Exception as e:
            logger.error(f"Error in Sonar analysis: {e}")
            return {
                'error': 'Failed to analyze findings with Sonar',
                'details': str(e),
                'query_sent': query if 'query' in locals() else None
            }
    
    async def _make_async_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async request to Perplexity API"""
        try:
            async with self.session.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                return await response.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"Async request error: {e}")
            raise
    
    async def _make_sync_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make synchronous request to Perplexity API"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    self.base_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=60
                )
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"Sync request error: {e}")
            raise
    
    def _process_response(
        self, 
        response_data: Dict[str, Any], 
        query: str, 
        model: str, 
        detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process Sonar API response"""
        try:
            # Extract main content
            choices = response_data.get('choices', [])
            if not choices:
                raise ValueError("No choices in response")
            
            message = choices[0].get('message', {})
            content = message.get('content', '')
            
            if not content:
                raise ValueError("No content in response")
            
            # Extract usage information
            usage = response_data.get('usage', {})
            
            # Parse structured response (attempt to extract sections)
            sections = self._parse_medical_response(content)
            
            return {
                'success': True,
                'analysis': {
                    'full_response': content,
                    'sections': sections,
                    'summary': self._generate_summary(sections, detections)
                },
                'metadata': {
                    'model_used': model,
                    'query_sent': query,
                    'response_length': len(content),
                    'usage': usage,
                    'findings_analyzed': len(detections)
                },
                'recommendations': self._extract_recommendations(content),
                'citations': self._extract_citations(content)
            }
            
        except Exception as e:
            logger.error(f"Error processing Sonar response: {e}")
            return {
                'error': 'Failed to process Sonar response',
                'details': str(e),
                'raw_response': response_data
            }
    
    def _parse_medical_response(self, content: str) -> Dict[str, str]:
        """Parse medical response into structured sections"""
        sections = {}
        
        # Common section headers to look for
        section_patterns = {
            'clinical_significance': ['clinical significance', 'significance', 'clinical importance'],
            'differential_diagnosis': ['differential diagnosis', 'differential', 'ddx'],
            'recommendations': ['recommendations', 'follow-up', 'next steps'],
            'clinical_correlation': ['clinical correlation', 'correlation', 'clinical context'],
            'research_evidence': ['research', 'evidence', 'literature', 'studies'],
            'limitations': ['limitations', 'considerations', 'caveats']
        }
        
        content_lower = content.lower()
        lines = content.split('\n')
        
        current_section = 'general'
        sections[current_section] = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = False
            for section_key, patterns in section_patterns.items():
                if any(pattern in line_lower for pattern in patterns):
                    current_section = section_key
                    sections[current_section] = []
                    section_found = True
                    break
            
            if not section_found and line.strip():
                sections[current_section].append(line.strip())
        
        # Convert lists to strings
        for key, value in sections.items():
            sections[key] = '\n'.join(value) if isinstance(value, list) else value
        
        return sections
    
    def _generate_summary(self, sections: Dict[str, str], detections: List[Dict[str, Any]]) -> str:
        """Generate a concise summary"""
        finding_count = len(detections)
        finding_types = set(d.get('class_name', 'Unknown') for d in detections)
        
        summary_parts = [
            f"Analysis of {finding_count} AI-detected findings",
            f"Finding types: {', '.join(list(finding_types)[:3])}" + ("..." if len(finding_types) > 3 else "")
        ]
        
        # Add key points from clinical significance
        clinical_sig = sections.get('clinical_significance', '')
        if clinical_sig:
            # Extract first sentence or key point
            first_sentence = clinical_sig.split('.')[0]
            if len(first_sentence) < 200:
                summary_parts.append(f"Key insight: {first_sentence}")
        
        return '. '.join(summary_parts) + '.'
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract actionable recommendations from response"""
        recommendations = []
        
        # Look for numbered lists, bullet points, or recommendation keywords
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for recommendation indicators
            rec_indicators = [
                'recommend', 'suggest', 'consider', 'should', 'follow-up',
                'next step', 'additional', 'further', 'repeat'
            ]
            
            if any(indicator in line.lower() for indicator in rec_indicators):
                # Clean up the line
                cleaned = line.lstrip('•-*123456789. ')
                if len(cleaned) > 10:  # Avoid very short fragments
                    recommendations.append(cleaned)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations and references from response"""
        citations = []
        
        # Look for common citation patterns
        import re
        
        # Pattern for journal citations
        journal_pattern = r'([A-Z][^.]*\.\s*[A-Z][^.]*\.\s*\d{4}[^.]*\.)'
        citations.extend(re.findall(journal_pattern, content))
        
        # Pattern for URLs
        url_pattern = r'(https?://[^\s]+)'
        citations.extend(re.findall(url_pattern, content))
        
        # Pattern for DOI
        doi_pattern = r'(doi:\s*[^\s]+)'
        citations.extend(re.findall(doi_pattern, content, re.IGNORECASE))
        
        return citations[:10]  # Limit to top 10 citations
    
    async def get_deep_research(
        self, 
        topic: str, 
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get deep research on a specific medical topic
        
        Args:
            topic: Medical topic to research
            context: Additional context for the research
        
        Returns:
            Dictionary containing research results
        """
        try:
            query = f"Provide a comprehensive medical research overview on: {topic}"
            if context:
                query += f"\n\nAdditional context: {context}"
            
            query += (
                "\n\nPlease include:\n"
                "- Current understanding and pathophysiology\n"
                "- Latest research findings and clinical trials\n"
                "- Diagnostic approaches and imaging findings\n"
                "- Treatment options and guidelines\n"
                "- Prognosis and outcomes\n"
                "- Recent publications and key references"
            )
            
            payload = {
                'model': self.models['sonar_large'],  # Use larger model for research
                'messages': [
                    {
                        'role': 'system',
                        'content': (
                            "You are a medical research specialist. Provide comprehensive, "
                            "evidence-based information with current research findings and "
                            "appropriate medical citations. Focus on recent developments "
                            "and clinical relevance."
                        )
                    },
                    {
                        'role': 'user',
                        'content': query
                    }
                ],
                'temperature': 0.2,  # Lower temperature for research
                'max_tokens': 2000   # More tokens for comprehensive research
            }
            
            # Make request
            if self.session:
                response_data = await self._make_async_request(payload)
            else:
                response_data = await self._make_sync_request(payload)
            
            return self._process_response(response_data, query, self.models['sonar_large'], [])
            
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return {
                'error': 'Failed to conduct deep research',
                'details': str(e),
                'topic': topic
            }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Perplexity API"""
        try:
            test_payload = {
                'model': self.default_model,
                'messages': [
                    {
                        'role': 'user',
                        'content': 'Hello, this is a connection test.'
                    }
                ],
                'max_tokens': 50
            }
            
            response = requests.post(
                self.base_url,
                headers=self._get_headers(),
                json=test_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'message': 'Connection successful',
                    'model': self.default_model
                }
            else:
                return {
                    'success': False,
                    'message': f'HTTP {response.status_code}: {response.text}',
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Connection failed: {str(e)}',
                'error': str(e)
            } 