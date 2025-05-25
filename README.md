# HealthLens - AI-Powered Medical Scan Analysis

HealthLens is an advanced medical imaging AI agent that combines computer vision models with expert consultation through Perplexity's Sonar API. It analyzes medical scans (2D and 3D) and provides intelligent insights to assist healthcare professionals.

## 🚀 Features

- **Multi-Modal Medical Imaging**: Support for 2D (X-rays, ultrasounds, mammography) and 3D (CT, MRI) scans
- **AI-Powered Detection**: Integration with YOLO, MedSAM, MONAI, and custom medical AI models
- **Expert Consultation**: Perplexity Sonar integration for research-backed medical insights
- **Comprehensive File Support**: DICOM, NIfTI, PNG, JPG, and more
- **RESTful API**: FastAPI-based backend with automatic documentation
- **Batch Processing**: Handle multiple scans simultaneously
- **Confidence Filtering**: Adjustable confidence thresholds for detections

## 📋 Supported Scan Types

### 2D Imaging

- Chest X-rays (`xray_chest_2d`)
- Abdominal X-rays (`xray_abdomen_2d`)
- Bone X-rays (`xray_bone_2d`)
- Ultrasound scans (`ultrasound_*_2d`)
- Mammography (`mammography_2d`)
- Fundus photography (`fundus_2d`)
- Dermatological images (`dermatology_2d`)
- Endoscopic images (`endoscopy_2d`)

### 3D Imaging

- CT scans (`ct_*_3d`)
- MRI scans (`mri_*_3d`)
- PET-CT (`pet_ct_3d`)
- SPECT (`spect_3d`)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)
- Perplexity API key

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd HealthLens
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Set up Perplexity API**

   - Get your API key from [Perplexity](https://www.perplexity.ai/)
   - Add it to your `.env` file:
     ```
     PERPLEXITY_API_KEY=your_api_key_here
     ```

5. **Add your models** (optional)
   - Place your trained models in the `models/` directory
   - Update model paths in `.env` if needed

## 🚀 Quick Start

### Start the API Server

```bash
# Development mode
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Basic Usage

#### Analyze a Single Scan

```bash
curl -X POST "http://localhost:8000/analyze_scan/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg" \
  -F "scan_type=xray_chest_2d" \
  -F "confidence_threshold=0.5" \
  -F "include_sonar_analysis=true"
```

#### Check Supported Scan Types

```bash
curl -X GET "http://localhost:8000/supported_scan_types"
```

#### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## 📊 API Endpoints

### Core Endpoints

| Endpoint                | Method | Description                      |
| ----------------------- | ------ | -------------------------------- |
| `/`                     | GET    | Health check                     |
| `/health`               | GET    | Detailed system status           |
| `/analyze_scan/`        | POST   | Analyze single medical scan      |
| `/analyze_batch/`       | POST   | Batch analysis of multiple scans |
| `/supported_scan_types` | GET    | List supported scan types        |
| `/model_info`           | GET    | Information about loaded models  |

### Request/Response Examples

#### Single Scan Analysis

**Request:**

```json
{
  "file": "<binary_data>",
  "scan_type": "xray_chest_2d",
  "confidence_threshold": 0.5,
  "include_sonar_analysis": true
}
```

**Response:**

```json
{
  "scan_type": "xray_chest_2d",
  "filename": "chest_xray.jpg",
  "processing_type": "2D",
  "vision_analysis": {
    "detections": [
      {
        "class_name": "Pneumonia",
        "confidence": 0.87,
        "bounding_box": [120, 150, 280, 320],
        "class_id": 1,
        "model_type": "yolo"
      }
    ],
    "statistics": {
      "total_detections": 1,
      "unique_classes": 1,
      "avg_confidence": 0.87
    },
    "processing_time": 1.23
  },
  "sonar_analysis": {
    "success": true,
    "analysis": {
      "full_response": "Based on the AI detection of pneumonia...",
      "summary": "Analysis of 1 AI-detected finding...",
      "sections": {
        "clinical_significance": "...",
        "recommendations": "..."
      }
    },
    "recommendations": [
      "Consider clinical correlation with patient symptoms",
      "Follow-up chest X-ray in 7-10 days"
    ],
    "citations": ["..."]
  }
}
```

## 🔧 Configuration

### Environment Variables

| Variable             | Description          | Default       |
| -------------------- | -------------------- | ------------- |
| `PERPLEXITY_API_KEY` | Perplexity API key   | Required      |
| `*_MODEL_PATH`       | Paths to model files | `models/*.pt` |
| `API_HOST`           | API host             | `0.0.0.0`     |
| `API_PORT`           | API port             | `8000`        |
| `LOG_LEVEL`          | Logging level        | `INFO`        |

### Model Integration

#### Adding Custom 2D Models

1. Place your model file in `models/`
2. Update the model path in `.env`
3. Modify `vision_processing/model_2d_handler.py` to load your model

#### Adding Custom 3D Models

1. Place your model file in `models/`
2. Update the model path in `.env`
3. Modify `vision_processing/model_3d_handler.py` to load your model

## 🏗️ Architecture

```
HealthLens/
├── main.py                 # FastAPI application
├── vision_processing/      # Computer vision modules
│   ├── common.py          # Shared utilities
│   ├── model_2d_handler.py # 2D model processing
│   └── model_3d_handler.py # 3D model processing
├── sonar_integration/      # Perplexity Sonar client
│   └── client.py          # API client implementation
├── models/                # Model storage
└── requirements.txt       # Dependencies
```

### Key Components

- **FastAPI Backend**: RESTful API with automatic documentation
- **Vision Processing**: Modular handlers for 2D and 3D medical imaging
- **Sonar Integration**: Intelligent medical consultation and research
- **Model Management**: Flexible model loading and configuration

## 🔬 Model Support

### Currently Supported

- **YOLO**: Ultralytics YOLOv8 for 2D detection
- **PyTorch**: Custom CNN models
- **MONAI**: Medical imaging framework (placeholder)

### Planned Support

- **MedSAM**: Medical Segment Anything Model
- **MedYOLO**: 3D medical object detection
- **Custom Transformers**: Vision transformers for medical imaging

## 🚨 Important Notes

### Medical Disclaimer

⚠️ **This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. All AI-generated findings should be reviewed by qualified medical professionals.**

### Data Privacy

- No medical data is stored permanently
- Temporary files are automatically cleaned up
- Consider HIPAA compliance for production use

### Performance Considerations

- GPU acceleration recommended for model inference
- Large 3D volumes may require significant memory
- Batch processing limited to prevent resource exhaustion

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the logs for debugging information

## 🔮 Roadmap

- [ ] Frontend web interface
- [ ] DICOM series support
- [ ] Real-time streaming analysis
- [ ] Multi-language support
- [ ] Integration with PACS systems
- [ ] Advanced visualization tools
- [ ] Model performance analytics

---

**Built for the future of medical AI** 🏥🤖
