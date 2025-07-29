# 🏙️ Toronto Airbnb Price Predictor

A Streamlit-powered web application that estimates the **nightly rental price** of a one-bedroom Airbnb listing in Toronto, using machine learning with XGBoost. Users can interactively input property details such as **neighbourhood**, **room type**, **bathrooms**, and more — and receive an instant predicted price, along with insights on how to optimize it.

🔗 **Live App:** [toronto-airbnb-price-predictor.streamlit.app](https://toronto-airbnb-price-predictor-rwszy2nnnij9qzdd9h7bmk.streamlit.app/)

---

## 📌 Features

- **💰 Price Estimation**  
  Predicts nightly Airbnb prices based on your property details using a trained XGBoost model.

- **📊 Visual Feedback**  
  Displays how your property compares to the city average (e.g., bathrooms, review scores, superhost status).

- **📈 Revenue Estimator**  
  Calculates potential **monthly earnings** based on occupancy rate.

- **📍 Neighbourhood & Room Type Impact**  
  Highlights how different locations or room types affect pricing (e.g., +$X for Waterfront, -$X for Private Room).

---

## 🧠 How It Works

### 1. Data Source & Modeling
- Sourced from the [Inside Airbnb](http://insideairbnb.com/)
- [Toronto dataset]([http://insideairbnb.com/](https://docs.google.com/spreadsheets/d/18DbatkTWogKwESjq4Hpe2pfvHOfto3DsnZwbu6o7cMM/edit?usp=sharing))
- Ingested and running on **capstone_code_modeling_hanna_lee.ipynb** which was initially written via Google Colab
- One-bedroom listings were filtered and cleaned to improve model performance.
- Outliers were removed and categorical variables (like neighbourhood cleansed, room type) were using **one-hot encoded**.
- Text-based descriptive data (like name, description, neighbourhood overview) were using **TF-IDF**

### 2. Model Training
- **capstone_code_modeling_hanna_lee.ipynb**
- Final Model: `XGBoost Regressor`
- Feature Engineering:
  - Categorical Encoding: Neighbourhood, Room Type, Property Type
  - Numerical Inputs: Bathrooms, Minimum Nights, Number of Reviews, Review Scores
- TF-IDF for textual data was tested but removed due to **worse R² scores** and **overfitting**
- Model and assets **(.joblib files)** stored to be used for predictor app

### 3. Deployment
- Frontend: Streamlit app
- App running on **predictor.py**
- Hosted on **Streamlit Cloud**
- Toronto Airbnb Price Predictor Launch
  
---
