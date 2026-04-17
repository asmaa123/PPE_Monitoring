# PPE Monitoring System
# Personal Protective Equipment Compliance Monitor

A professional AI-powered PPE monitoring system built with Streamlit and YOLOv8 for real-time detection and compliance tracking.

## 🚀 Features

- **Real-time PPE Detection**: YOLOv8 model detects 8 PPE classes (Gloves, Helmet, No-Gloves, No-Helmet, No-Shoes, No-Vest, Shoes, Vest)
- **Position-based Filtering**: Intelligent filtering ignores helmet detections in upper frame areas and shoe detections in lower areas
- **Video Processing**: Upload videos for batch processing with annotated results
- **Live Webcam**: Real-time monitoring from webcam feed
- **Analytics Dashboard**: Interactive charts showing compliance rates and violation statistics
- **Database Integration**: SQLite database for storing violation logs and analytics
- **Professional UI**: Modern dark theme with gradient styling and animations

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Data Visualization**: Plotly
- **Database**: SQLite
- **Deployment**: Streamlit Cloud

## 📋 Requirements

- Python 3.8+
- YOLOv8 model weights (best.pt)
- Required packages listed in requirements.txt

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ppe-monitoring.git
cd ppe-monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
ppe-monitoring/
├── app.py                 # Main Streamlit application
├── detector.py           # PPE detection logic with YOLOv8
├── database.py           # SQLite database operations
├── requirements.txt      # Python dependencies
├── models/
│   └── best.pt          # YOLOv8 model weights
├── screenshots/          # App screenshots
└── README.md            # This file
```

## 🎯 Usage

1. **Model Setup**: Place your trained YOLOv8 model (best.pt) in the `models/` directory
2. **Launch App**: Run `streamlit run app.py`
3. **Upload Video**: Use the file uploader to process videos
4. **Live Monitoring**: Switch to webcam mode for real-time detection
5. **View Analytics**: Check compliance statistics and violation trends

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit framework
- OpenCV for computer vision