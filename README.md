# Jaundice Detection using CNN & HOG

A non-invasive neonatal jaundice detection system using Deep Learning and Image Processing.

## 📌 About the Project

Neonatal jaundice affects 60% of full-term and 80% of preterm newborns. Traditional diagnosis requires invasive blood tests. This system detects jaundice non-invasively by analyzing facial images of newborns using Deep Learning.

Built as a B.Tech final year project at Jansons Institute of Technology, Coimbatore.
Published in IJRAR Journal — Volume 12, Issue 2, April 2025.
Presented at ICSTEM 4.25 International Conference, March 2025.

## 🎯 How It Works

1. Upload a neonatal facial image
2. System detects the forehead region (ROI) using Haar Cascade
3. Extracts color features using RGB and YCbCr color spaces
4. HOG algorithm extracts texture features
5. DenseNet121 CNN classifies as Jaundiced or Normal
6. Result displayed with confidence score

## 🏆 Results

- Model Accuracy: 92%
- Architecture: DenseNet121 (Transfer Learning)
- Deployed as: Streamlit Web Application

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- DenseNet121
- Streamlit
- NumPy, Pandas, Matplotlib

## 📁 Project Structure



Jaundice-Detection-CNN/
│
├── backend.py       # Model training, HOG feature extraction, CNN architecture
├── app.py           # Streamlit frontend for real-time prediction
└── CNNMODEL.h5      # Trained model (not uploaded due to size)



## 🚀 How to Run

```bash
# Install dependencies
pip install tensorflow opencv-python streamlit numpy pandas scikit-learn

# Run the app
streamlit run app.py
```

## 👥 Team

- G. Bindubhavya
- S. Sarvetha
- N.B. Nithishwaran
- K. Kevin

Supervisor: Mrs. T. Ramya, Assistant Professor, Dept. of AI & DS, Jansons Institute of Technology

## 📜 Publication

- Journal: Deep Learning Based – Non Intrusive Detection of Jaundice in Neonatals  
  IJRAR, Volume 12, Issue 2, April 2025
- Conference: ICSTEM 4.25, Jansons Institute of Technology, March 2025

## 📄 Documents

- [Thesis Report](https://www.dropbox.com/scl/fi/n8j4t07xmlf5r7jvfcmuv/Final-Thesis-DOC.pdf?rlkey=yj5stw4kg6mo2q3kl4sap7ods&st=als4m7g6&dl=0)
- [Published Paper](https://www.dropbox.com/scl/fi/vs8tozo5of0z1t7rdld81/Deep-Learning-Based-Non-Intrusive-Detection-Of-Jaundice-In-Neonatals.pdf?rlkey=z5v63c938bimnzkxbrst8hl8z&st=za1xuf2e&dl=0)

## 📬 Contact

G. Bindubhavya 
gbindubhavya16@gmail.com  
[LinkedIn](https://www.linkedin.com/in/gollavilli-bindubhavya/)
