# 🌧️ Rainfall Prediction and 7-Day Weather Forecast System

## 📌 Project Overview
This project is a Machine Learning–based Rainfall Prediction System that automatically detects the user's geographical location and predicts:

- Whether it will rain (Yes/No)
- Expected rainfall amount (in mm)
- 7-Day rainfall forecast

The system collects historical weather data using the Open-Meteo API and applies Machine Learning models to generate rainfall predictions.

---

## 🚀 Features
- Automatic location detection using IP Geolocation
- Manual latitude & longitude override option
- Historical weather data collection (2020–2024)
- Rain prediction using Machine Learning
- Rainfall amount estimation
- 7-Day future rainfall forecast
- Model performance evaluation metrics
- Command-line based execution

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries
- NumPy
- Pandas
- Scikit-learn
- Requests
- Argparse
- Datetime

### Machine Learning Models
- Random Forest Classifier (Rain Prediction)
- Random Forest Regressor (Rainfall Amount Prediction)

### Data Source
- Open-Meteo Weather API

---

## 🧠 Machine Learning Workflow

### 1. Data Collection
Historical weather parameters collected:
- Maximum Temperature
- Minimum Temperature
- Relative Humidity
- Surface Pressure
- Wind Speed
- Precipitation

Date Range:
2020-01-01 to 2024-12-31

---

### 2. Feature Engineering
Additional features generated:
- Previous day precipitation
- Previous day temperature
- Previous day humidity
- Month
- Day of Year
- Rain classification label

Rain Threshold:
Precipitation > 0.1 mm → Rain = YES

---

### 3. Model Training

#### Classification Model
Predicts rainfall occurrence:
Rain → YES / NO

Evaluation Metrics:
- Accuracy
- Precision
- Recall
- F1 Score

#### Regression Model
Predicts rainfall quantity:
Rainfall Amount (mm)

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## ⚙️ Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/rainfall-prediction.git
cd rainfall-prediction
```

### Step 2: Install Required Libraries
```bash
pip install pandas numpy scikit-learn requests
```

---

## ▶️ How to Run the Project

### Automatic Location Detection
```bash
python main.py
```

### Manual Latitude & Longitude
```bash
python main.py --lat 16.5062 --lon 80.6480
```

### Disable Auto Detection
```bash
python main.py --no-auto
```

---

## 📊 Output Example
```
Model Performance Metrics:

Classification Model:
Accuracy: 92.45%
Precision: 0.91
Recall: 0.89
F1-Score: 0.90

========================================================
7-DAY RAINFALL FORECAST
========================================================

2026-02-28 (Sat)
Rain: YES
Predicted Rainfall: 4.321 mm
```

---

## 📂 Project Structure
```
rainfall-prediction/
│
├── main.py
├── README.md
└── requirements.txt
```

---

## 🌍 Location Detection Method
The system attempts automatic location detection using:
- ipinfo.io
- ipwho.is
- ip-api
- geoplugin

If automatic detection fails, manual coordinate input is requested.

---

## 🎯 Applications
- Smart Agriculture
- Irrigation Planning
- Weather Analysis
- Flood Prediction Support
- Water Resource Management

---

## 🔮 Future Enhancements
- Web Dashboard Integration
- Data Visualization Graphs
- Deep Learning Models
- Mobile Application Support
- IoT Weather Sensor Integration

---

## 👩‍💻 Author
**Mounika Chowdary**  
B.Tech – Computer Science and Engineering 
SRM University AP

**Pavithra Pamula**  
B.Tech – Computer Science and Engineering  
SRM University AP  

**Akhila Chirumamilla**  
B.Tech – Computer Science and Engineering  
SRM University AP  

**Asritha Asireddy**  
B.Tech – Computer Science and Engineering 
SRM University AP  

**K Kavya**  
B.Tech – Computer Science and Engineering  
SRM University AP  

---

## 📜 License
This project is developed for academic and research purposes only.
