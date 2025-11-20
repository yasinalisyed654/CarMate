# 🚗 CarMate  
### *Your Smart Used Car Advisor*

---

## 🧩 Abstract

In today’s used vehicle market, finding the right car that fits a user’s preferences—brand, model, fuel type, or budget—is a challenging task.  
This project presents a **content-based car recommendation system** that helps users discover cars similar to their selected car or query input using **unsupervised machine learning**.  

The dataset used in this project is derived from the **[Quickr Cars Dataset](https://www.kaggle.com/datasets/vedantkhapekar/quickr-cars-dataset)** available on Kaggle, which contains detailed listings of used cars sold across India.  
After cleaning and preprocessing, the data was used to build and deploy a **Car Recommendation Engine** powered by **k-Nearest Neighbors (k-NN)** and **Cosine Similarity**.  

The final product is an interactive **Streamlit web application** that enables users to filter, search, and find the most similar cars within their criteria.  
This project demonstrates the application of machine learning in **personalized recommendations** and the **used automobile marketplace**.

---

## 🎯 Objectives

- Build a **content-based recommendation system** for used cars.  
- Allow users to find cars similar to a specific model or keyword query.  
- Apply **TF-IDF Vectorization** and **k-Nearest Neighbors** for similarity measurement.  
- Develop an interactive, user-friendly interface using **Streamlit**.  
- Provide users with practical insights and an intuitive exploration experience.

---

## 🧠 Dataset

**Source:** [Quickr Cars Dataset on Kaggle](https://www.kaggle.com/datasets/vedantkhapekar/quickr-cars-dataset)  
**Cleaned Version Used:** [`Cleaned_Car_data.csv`](https://github.com/rajtilakls2510/car_price_predictor/blob/master/Cleaned_Car_data.csv)

**Key Columns:**
| Column | Description |
|--------|--------------|
| `name` | Car model name |
| `company` | Manufacturer (e.g., Maruti, Hyundai, Ford) |
| `year` | Year of manufacture |
| `Price` | Selling price (for display, not prediction) |
| `kms_driven` | Total kilometers driven |
| `fuel_type` | Type of fuel (Petrol/Diesel/CNG) |

---

## ⚙️ System Overview

### 🔹 1. Recommendation Engine
- **Algorithm:** k-Nearest Neighbors (k-NN)  
- **Similarity Metric:** Cosine Similarity  
- **Goal:** Recommend similar cars based on a selected car or user query.  
- **Approach:**  
  - Combine text features (`name`, `company`, `fuel_type`) using **TF-IDF vectorization**.  
  - Normalize numerical features (`year`, `kms_driven`) using **StandardScaler**.  
  - Merge text and numeric vectors into a hybrid feature space.  
  - Use **NearestNeighbors(metric='cosine')** to find the top *k* most similar cars.

### 🔹 2. Streamlit Web App
- Simple and responsive user interface.  
- Two main modes:
  1. **Find Similar Cars:** Select a car from the list to get top similar ones.  
  2. **Search by Text Query:** Enter a phrase like “Diesel SUV Honda” to discover matching cars.  
- Interactive sidebar filters for:
  - Price range  
  - Company  
  - Fuel type  
  - Year range  
  - Max kilometers driven  

---

## 🧰 Tech Stack

| Layer | Tools / Libraries |
|--------|-------------------|
| Frontend | Streamlit |
| Backend | Python |
| Machine Learning | scikit-learn |
| Data Handling | Pandas, NumPy |
| Feature Engineering | TF-IDF Vectorizer, StandardScaler |
| Similarity Metric | Cosine Similarity |
| Dataset | Quickr Cars Dataset (Kaggle) |

---

## 📊 Methodology

1. **Data Cleaning & Preprocessing**  
   - Removed missing or inconsistent entries.  
   - Converted text fields to lowercase and stripped whitespace.  
   - Combined multiple text columns into a single searchable string.  

2. **Feature Engineering**  
   - Applied TF-IDF on text columns (`name`, `company`, `fuel_type`).  
   - Scaled numerical columns (`year`, `kms_driven`) using StandardScaler.  
   - Merged text and numeric vectors for each car record.  

3. **Model Development**  
   - Used k-Nearest Neighbors with cosine distance for similarity measurement.  
   - Built two recommendation modes: *based on car selection* and *based on user text query*.  

4. **Deployment**  
   - Developed an interactive interface with Streamlit.  
   - Integrated filters for flexible user exploration.  

---

## 🚀 How to Run the Project

..........
..........
..........

