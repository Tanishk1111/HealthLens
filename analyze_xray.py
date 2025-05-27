#!/usr/bin/env python3
"""
X-ray Analysis Script for HealthLens

This script allows you to easily analyze X-ray images and get diagnosis
results with Sonar integration.
"""

import requests
import json
import sys
from pathlib import Path

def analyze_xray(image_path, scan_type="xray_chest_pathology_2d", confidence_threshold=0.1):
    """
    Analyze an X-ray image using HealthLens API with Sonar integration
    
    Args:
        image_path (str): Path to your X-ray image file
        scan_type (str): Type of analysis to perform
        confidence_threshold (float): Minimum confidence for detections
    """
    
    # Check if image file exists
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    # API endpoint
    url = "http://localhost:8000/analyze_scan/"
    
    print(f"🔍 Analyzing X-ray image: {image_file.name}")
    print(f"📋 Scan type: {scan_type}")
    print(f"🎯 Confidence threshold: {confidence_threshold}")
    print("⏳ Processing...")
    
    try:
        # Prepare the request
        with open(image_file, 'rb') as f:
            files = {
                'file': (image_file.name, f, 'image/jpeg')
            }
            data = {
                'scan_type': scan_type,
                'confidence_threshold': confidence_threshold
            }
            
            # Send request to API
            response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis Complete!")
            print("=" * 60)
            
            # Display vision analysis results
            vision_analysis = result.get('vision_analysis', {})
            detections = vision_analysis.get('detections', [])
            
            print(f"📊 VISION ANALYSIS RESULTS:")
            print(f"   • Processing time: {vision_analysis.get('processing_time', 'N/A')} seconds")
            print(f"   • Total detections: {len(detections)}")
            
            if detections:
                print(f"   • Detected conditions:")
                for i, detection in enumerate(detections, 1):
                    class_name = detection.get('class_name', 'Unknown')
                    confidence = detection.get('confidence', 0)
                    print(f"     {i}. {class_name} (confidence: {confidence:.2%})")
            else:
                print(f"   • No significant pathologies detected")
            
            # Display Sonar analysis results
            sonar_explanation = result.get('sonar_llm_explanation', '')
            sonar_query = result.get('sonar_query_sent_to_llm', '')
            
            print(f"\n🩺 SONAR EXPERT ANALYSIS:")
            if sonar_explanation and sonar_explanation != "Sonar analysis not performed for this request.":
                print(f"   Query sent to Sonar LLM:")
                print(f"   {sonar_query}")
                print(f"\n   Expert Analysis:")
                print(f"   {sonar_explanation}")
            else:
                print(f"   {sonar_explanation}")
                if "API key missing" in sonar_explanation:
                    print(f"   💡 To enable Sonar analysis, add your Perplexity API key to .env file")
            
            print("=" * 60)
            
            return result
            
        else:
            print(f"❌ Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to HealthLens API")
        print("Make sure the server is running with: python start_server.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function to handle command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_xray.py <path_to_xray_image>")
        print("\nExample:")
        print("  python analyze_xray.py chest_xray.jpg")
        print("  python analyze_xray.py /path/to/your/xray.png")
        print("\nSupported scan types:")
        print("  - xray_chest_pathology_2d (default)")
        print("  - xray_abdomen_2d")
        print("  - xray_bone_2d")
        sys.exit(1)
    
    image_path = sys.argv[1]
    scan_type = sys.argv[2] if len(sys.argv) > 2 else "xray_chest_pathology_2d"
    confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    # Analyze the X-ray
    result = analyze_xray(image_path, scan_type, confidence_threshold)
    
    if result:
        # Optionally save results to file
        output_file = f"analysis_results_{Path(image_path).stem}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main() 