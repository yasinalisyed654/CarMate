# 🚗 CarMate  
### *Your Smart Used Car Advisor*

---

## 🧩 Abstract

In today’s pre-owned vehicle market, finding the right car that fits user preferences—brand, model, fuel type, mileage, and budget—is a challenging task.  
This project, **CarMate**, is a **content-based car recommendation system** that helps users explore cars similar to a selected model using **machine learning techniques**.  

The system uses the **[Quickr Cars Dataset](https://www.kaggle.com/datasets/vedantkhapekar/quickr-cars-dataset)** as its base dataset and applies data cleaning, preprocessing, and vectorization to build a robust recommendation engine.  
It combines **TF-IDF (Term Frequency–Inverse Document Frequency)** for text data and **k-Nearest Neighbors (k-NN)** for similarity matching.  

The final product is an interactive **Streamlit web application** that enables users to:
- Choose a car brand and model,
- Set filters (year, fuel type, mileage, and price),
- And instantly get recommendations for similar cars.  

This project demonstrates the integration of **machine learning, data preprocessing, and web deployment** to solve a real-world problem in the used automobile industry.

---

## 🎯 Objectives

- Build a **content-based recommendation system** for used cars.  
- Allow users to search for cars similar to a specific model or query.  
- Apply **TF-IDF vectorization** on textual data and **k-NN** for similarity detection.  
- Develop an **interactive, user-friendly interface** with Streamlit.  
- Integrate **filter-based constraints** (price, fuel, year, mileage) dynamically.  
- Demonstrate practical application of ML for **personalized recommendations**.  

---

## 🧠 Dataset

**Source:** [Quickr Cars Dataset – Kaggle](https://www.kaggle.com/datasets/vedantkhapekar/quickr-cars-dataset)  

**Key Columns:**

| Column | Description |
|--------|--------------|
| `name` | Car model name |
| `company` | Manufacturer (e.g., Maruti, Hyundai, Ford) |
| `year` | Manufacturing year |
| `Price` | Selling price (used for filtering, not prediction) |
| `kms_driven` | Total kilometers driven |
| `fuel_type` | Fuel type (Petrol/Diesel/CNG/etc.) |

---

## ⚙️ System Overview

### 🔹 1. Recommendation Engine
- **Algorithm:** k-Nearest Neighbors (k-NN)  
- **Similarity Metric:** Cosine Similarity  
- **Text Processing:** TF-IDF vectorization of `name`, `company`, and `fuel_type`.  
- **Numeric Features:** `year` and `kms_driven` scaled using `StandardScaler`.  
- **Goal:** Recommend the *Top-K* most similar cars to a selected model or query.  

### 🔹 2. Streamlit Web Application
- Interactive web app with no sidebar clutter.  
- Flow:
  1. **Select Brand → Model** (filtered dynamically).  
  2. Choose **Fuel Type** and number of results.  
  3. Set **Price range**, **Year range**, and **Max mileage (kms)**.  
  4. Click **Find Recommendations** to view top results.  
- The app automatically:
  - Adapts slider bounds to brand/model selection.  
  - Handles missing or single-value data gracefully.  
  - Soft-relaxes overly strict filters to ensure valid results.  

---

## 🧰 Tech Stack

| Component | Tools / Libraries |
|------------|-------------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Python, scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Feature Engineering** | TF-IDF Vectorizer, StandardScaler |
| **Model** | k-Nearest Neighbors (cosine similarity) |
| **Deployment** | Streamlit App |

---

## 📊 Methodology

1. **Data Cleaning & Preprocessing**
   - Removed missing, duplicated, or inconsistent rows.
   - Normalized text columns (lowercase, stripped whitespace).
   - Converted numerical columns (`Price`, `kms_driven`, `year`) to proper numeric types.

2. **Feature Engineering**
   - Combined textual columns into a single string for vectorization.
   - Used TF-IDF to transform text data into numerical feature vectors.
   - Scaled numeric features for equal contribution to similarity computation.

3. **Model Development**
   - Built a **k-NN model** using cosine similarity.
   - Precomputed vectors for all cars in the dataset.
   - Queried the model for the most similar cars given user inputs.

4. **Web Integration**
   - Implemented an interactive Streamlit UI:
     - Dynamic sliders adjust based on selected brand/model data.
     - Auto-handles one-value columns (like a single price/year entry).
   - Provided clear user messages for missing or relaxed filters.

---

## 🧮 Inputs & Outputs

### **Inputs**
- Brand  
- Model (filtered by brand)  
- Fuel type  
- Price range  
- Year range  
- Max mileage (kms)  
- Number of recommendations  

### **Outputs**
A ranked table of **Top-K recommended cars**, including:
| Score | Name | Company | Fuel | Year | Kms Driven | Price |
|--------|------|----------|------|------|-------------|--------|

Scores represent similarity (1 = most similar).

---

## 🚀 How to Run the Project

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/<your-username>/carmate.git
cd carmate
