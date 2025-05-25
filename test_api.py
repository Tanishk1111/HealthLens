#!/usr/bin/env python3
"""
Test script for HealthLens API

This script tests the basic functionality of the HealthLens API
without requiring actual medical images or API keys.
"""

import requests
import json
import time
from io import BytesIO
from PIL import Image
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image for testing"""
    # Create a 512x512 RGB test image
    image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Convert to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_detailed_health():
    """Test the detailed health endpoint"""
    print("\n🔍 Testing detailed health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Detailed health check failed: {e}")
        return False

def test_supported_scan_types():
    """Test the supported scan types endpoint"""
    print("\n🔍 Testing supported scan types...")
    try:
        response = requests.get(f"{BASE_URL}/supported_scan_types")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"2D types: {len(data.get('2d_types', {}))}")
        print(f"3D types: {len(data.get('3d_types', {}))}")
        print(f"Total supported: {len(data.get('supported_scan_types', {}))}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Supported scan types test failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"2D models: {list(data.get('2d_models', {}).keys())}")
        print(f"3D models: {list(data.get('3d_models', {}).keys())}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

def test_2d_scan_analysis():
    """Test 2D scan analysis"""
    print("\n🔍 Testing 2D scan analysis...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Prepare request
        files = {
            'file': ('test_xray.png', test_image, 'image/png')
        }
        data = {
            'scan_type': 'xray_chest_2d',
            'confidence_threshold': 0.5,
            'include_sonar_analysis': False  # Disable to avoid API key requirement
        }
        
        print("Sending request...")
        response = requests.post(f"{BASE_URL}/analyze_scan/", files=files, data=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Processing type: {result.get('processing_type')}")
            print(f"Detections: {len(result.get('vision_analysis', {}).get('detections', []))}")
            print(f"Processing time: {result.get('vision_analysis', {}).get('processing_time')}s")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 2D scan analysis test failed: {e}")
        return False

def test_3d_scan_analysis():
    """Test 3D scan analysis (with placeholder data)"""
    print("\n🔍 Testing 3D scan analysis...")
    try:
        # Create a simple test file (not a real NIfTI, but for testing)
        test_data = b"fake_nifti_data_for_testing" * 100
        
        files = {
            'file': ('test_ct.nii.gz', test_data, 'application/octet-stream')
        }
        data = {
            'scan_type': 'ct_chest_3d',
            'confidence_threshold': 0.5,
            'include_sonar_analysis': False
        }
        
        print("Sending request...")
        response = requests.post(f"{BASE_URL}/analyze_scan/", files=files, data=data)
        print(f"Status: {response.status_code}")
        
        if response.status_code in [200, 400]:  # 400 expected for fake data
            if response.status_code == 400:
                print("Expected error for fake 3D data (this is normal)")
            else:
                result = response.json()
                print(f"Processing type: {result.get('processing_type')}")
            return True
        else:
            print(f"Unexpected error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 3D scan analysis test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting HealthLens API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Detailed Health", test_detailed_health),
        ("Supported Scan Types", test_supported_scan_types),
        ("Model Info", test_model_info),
        ("2D Scan Analysis", test_2d_scan_analysis),
        ("3D Scan Analysis", test_3d_scan_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        
        results.append({
            'name': test_name,
            'success': success,
            'duration': duration
        })
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['name']:<25} ({result['duration']:.2f}s)")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HealthLens API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and configuration.")
    
    return passed == total

if __name__ == "__main__":
    print("HealthLens API Test Suite")
    print("Make sure the API server is running on http://localhost:8000")
    print("Start the server with: python main.py")
    print()
    
    input("Press Enter to start tests...")
    
    success = run_all_tests()
    exit(0 if success else 1) 