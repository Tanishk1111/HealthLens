import requests
import json

def analyze_xray(image_path, server_url="http://127.0.0.1:8000"):
    """
    Send X-ray image to HealthLens for analysis
    """
    # Prepare the files and form data
    with open(image_path, 'rb') as image_file:
        files = {
            'file': ('xray.png', image_file, 'image/png')
        }
        data = {
            'scan_type': 'xray_chest_pathology_2d',
            'confidence_threshold': 0.1
        }
        
        print(f"🔍 Sending X-ray image to HealthLens...")
        print(f"📁 Image: {image_path}")
        print(f"🔗 Server: {server_url}/analyze_scan/")
        
        try:
            # Send POST request
            response = requests.post(
                f"{server_url}/analyze_scan/", 
                files=files, 
                data=data,
                timeout=120  # 2 minutes timeout for AI processing
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\n✅ SUCCESS! Analysis completed!")
                print("="*50)
                
                # Display vision analysis results
                vision_analysis = result.get('vision_analysis', {})
                detections = vision_analysis.get('detections', [])
                
                print(f"🔬 VISION ANALYSIS RESULTS:")
                if detections:
                    print(f"📊 Found {len(detections)} pathology predictions:")
                    for i, detection in enumerate(detections[:5], 1):  # Show top 5
                        name = detection.get('class_name', 'Unknown')
                        confidence = detection.get('confidence', 0)
                        print(f"   {i}. {name}: {confidence:.1%} confidence")
                else:
                    print("   No significant pathologies detected above threshold")
                
                # Display Sonar insights
                sonar_explanation = result.get('sonar_llm_explanation', '')
                print(f"\n🧠 AI MEDICAL INSIGHTS:")
                print("-" * 30)
                if sonar_explanation and sonar_explanation != "Sonar analysis not performed for this request.":
                    print(sonar_explanation)
                else:
                    print("No AI insights generated (check Perplexity API key)")
                
                print("\n" + "="*50)
                print("⚠️  DISCLAIMER: This is AI analysis for informational purposes only.")
                print("   Always consult healthcare professionals for medical decisions.")
                
                return result
                
            else:
                print(f"❌ ERROR: Server returned status {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
            return None

# Usage
if __name__ == "__main__":
    # CHANGE THIS PATH to your actual X-ray image
    xray_path = r"C:\Users\ASUS\Desktop\COCO\HealthLens\Xray.jpeg"  # UPDATE THIS!
    
    print("🏥 HealthLens X-ray Analysis")
    print("="*30)
    
    result = analyze_xray(xray_path)
    
    if result:
        # Optionally save the full result to a JSON file
        with open('xray_analysis_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Full results saved to: xray_analysis_result.json")
