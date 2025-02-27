# 💰 Financial Health Prediction 📊

**Deployed using FastAPI on Render:**
🔗 [ Financial Health Prediction ](https://machine-learning-13.onrender.com/docs)
---

## 🌟 Overview
Welcome to the **Financial Health Prediction** project! 🚀 This machine learning model predicts an individual's **financial health** based on key financial features such as **income, savings, expenditure, debt, and investment**.

The model utilizes a **Random Forest Classifier** and is deployed as a **FastAPI application** for real-time predictions. 🔥

### 🧐 Why is this Important?
Financial health is critical for long-term well-being. This project helps users evaluate their financial situation and make informed decisions for financial stability. 💡🔒

---
## 🔑 Key Features
✅ **Data Exploration**: Visualize and understand relationships in the dataset. 📊  
✅ **Machine Learning Model**: Predict financial health using **Random Forest Classifier**. 🤖  
✅ **Model Evaluation**: Assess performance using **accuracy, classification reports, and confusion matrix**. 📈  
✅ **Real-time API**: Make financial health predictions with real-time input via **FastAPI**. ⚡  
✅ **Scalability**: Easily extendable with more features or complex models. 🌱

---
## ⚙️ How It Works

### 1️⃣ Data Exploration
We start by **analyzing the dataset**, visualizing feature relationships, and understanding patterns in **financial health**.

### 2️⃣ Data Preprocessing
🔹 Splitting the dataset into **features** and **target variables**.  
🔹 Dividing data into **training (80%)** and **testing (20%)** sets.  
🔹 Scaling numerical features using **StandardScaler** for optimal performance.  

### 3️⃣ Model Training
We train a **Random Forest Classifier**, a powerful and flexible ensemble learning model, for financial health prediction.  

### 4️⃣ Model Evaluation
📌 **Accuracy** - Overall model accuracy.  
📌 **Classification Report** - Precision, recall, and F1-score for each class.  
📌 **Confusion Matrix** - Visualizing the model’s performance.  

### 5️⃣ Model Deployment
The trained model is deployed as a **FastAPI application**, allowing users to interact with the model through a **RESTful API**. 🌍

---
## 🔧 Requirements
To run this project locally, install the following dependencies:
```bash
pip install -r requirements.txt
```
### Required Packages:
- Python 3.7+ 🐍  
- FastAPI 🌐  
- Uvicorn ⚡  
- Pandas 🐼  
- NumPy 🔢  
- Scikit-learn 🧠  
- Matplotlib 📊  
- Seaborn 🌈  
- Joblib 📦  
- Pydantic 🧳  

---
## 🔍 How to Use
### 1️⃣ Clone the Repository
```bash
git clone https:https://github.com/abigiyaelias20/Machine-learning.git
cd financial-health-prediction
```
### 2️⃣ Run the FastAPI Application
```bash
uvicorn api:app --reload
```
This runs the server at **http://127.0.0.1:8000** 🌍

### 3️⃣ Make Predictions
Use **Postman** or **cURL** to send a **POST request** to the `/predict` endpoint.

#### Example Request Body (JSON format):
```json
{
    "income": 80000,
    "savings": 20000,
    "expenditure": 25000,
    "debt": 2000,
    "investment": 15000
}
```

#### Example Response:
```json
{
  "financial_health_prediction": o
}
```
```json
{
    "income": 20000,
    "savings": 1000,
    "expenditure": 15000,
    "debt": 7000,
    "investment": 500
}
```

#### Example Response:
```json
{
  "financial_health_prediction": 1
}
```


---
## 📚 Model Details
- **Model Used**: Random Forest Classifier (100 estimators)
- **Evaluation Metrics**:
  - Accuracy: Overall model accuracy.
  - Precision, Recall, F1-score: Key metrics for performance.
  - Confusion Matrix: Visualizing true positives, false positives, etc.

---
## 📁 Project Files
📂 `financial_data_large.csv` - Dataset with financial attributes and the target variable `financial_health`.  
📂 `financial_health_model.pkl` - Trained **Random Forest model** stored as a pickle file.  
📂 `scaler.pkl` - StandardScaler model used for feature scaling.  
📂 `api.py` - FastAPI application containing the `/predict` endpoint.  

---
## 🚀 Future Improvements
🔧 **Enhanced Features** - Add additional features such as **credit score, employment history**, etc.  
⚙️ **Model Tuning** - Experiment with **hyperparameter tuning** to improve model performance.  
💻 **User Interface** - Develop a **web-based UI** for easy user interaction.  

---
## 🤝 Contributing
Contributions are always welcome! 🎉  
If you have ideas to improve the project, feel free to **fork**, create an **issue**, or submit a **pull request**.  
Together, we can make this project even better. 💪

## 📧 Contact
For any questions or inquiries about the project, reach out to:
📩 Email:abigiyaelias709@gmail.com 
💻 GitHub: [@yourusername](https://github.com/abigiyaelias20)  

🚀 **Let’s build financial awareness together!** 🚀




