# 🔢 Handwritten Digit Recognition

A machine learning project that recognizes handwritten digits from 0 to 9 using Python, Scikit-learn, the MNIST dataset, and Streamlit.

## 🚀 Features

- **Single-digit recognition:** Upload an image containing one handwritten digit.
- **10-digit grid recognition:** Predict digits from a regular grid containing 2 rows and 5 columns.
- **Prediction probabilities:** View the model's estimated probability for each digit.
- **Interactive web interface:** Use the application through a Streamlit dashboard.
- **Model evaluation:** Includes accuracy metrics and a confusion matrix.

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Pillow
- Joblib
- Streamlit

## 📂 Project Structure

```text
Handwritten-Recognition/
├── data/
├── models/
│   ├── model.joblib
│   └── preprocessing.joblib
├── outputs/
│   ├── confusion_matrix.png
│   └── model_metrics.csv
├── src/
│   └── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ramk09/CodeAlpha_HandwrittenRecognition.git
cd CodeAlpha_HandwrittenRecognition
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## 🧠 Model and Dataset

The model is a Scikit-learn `MLPClassifier` trained using the MNIST handwritten-digit dataset.

The model accepts 28 × 28 grayscale pixel inputs and predicts one of ten digit classes, from 0 through 9.

To retrain the model, run:

```bash
python src/train.py
```

The training script downloads the MNIST dataset when needed and saves the trained model and evaluation outputs.

## 🖼️ Application Screenshots

Screenshots can be added here after finalizing the interface.

## ⚠️ Limitations

- Single-digit mode works best with one clearly visible digit.
- Grid mode expects a regular layout of exactly 2 rows and 5 columns.
- Grid mode divides the image into equal-sized cells; it does not automatically detect character boundaries.
- Prediction probabilities are model estimates and do not guarantee correctness.

## 👨‍💻 Author

**Kurra Venkata Siva Rama Krishna

  Machine Learning**

GitHub: [@Ramk09](https://github.com/Ramk09)

## 📄 License

No license has been added yet. Contact the repository owner for permission before reusing this project.
