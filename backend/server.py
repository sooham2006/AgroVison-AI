from flask import Flask, request, jsonify
from flask_cors import CORS
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# --- Crop Analysis Section (from crop-analysis/server.py) ---
# Load crop production data
crop_df = pd.read_csv('India-Agriculture-Crop-Production.csv')

# Standard crop norms
crop_norms = {
    'Rice': {'yield_optimal': 3.5, 'water_mm': 1200, 'N_kg': 100, 'P_kg': 40, 'K_kg': 80},
    'Arecanut': {'yield_optimal': 2.5, 'water_mm': 1200, 'N_kg': 100, 'P_kg': 40, 'K_kg': 150},
    'Banana': {'yield_optimal': 20.0, 'water_mm': 1800, 'N_kg': 200, 'P_kg': 60, 'K_kg': 300},
    'Black pepper': {'yield_optimal': 0.5, 'water_mm': 1200, 'N_kg': 150, 'P_kg': 50, 'K_kg': 150},
    'Cashewnut': {'yield_optimal': 1.0, 'water_mm': 800, 'N_kg': 80, 'P_kg': 40, 'K_kg': 60},
    'Coconut': {'yield_optimal': 10.0, 'water_mm': 1500, 'N_kg': 120, 'P_kg': 60, 'K_kg': 180},
    'Dry chillies': {'yield_optimal': 1.5, 'water_mm': 600, 'N_kg': 100, 'P_kg': 50, 'K_kg': 50},
    'Ginger': {'yield_optimal': 15.0, 'water_mm': 1300, 'N_kg': 120, 'P_kg': 60, 'K_kg': 100},
    'Other Kharif pulses': {'yield_optimal': 1.0, 'water_mm': 500, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'other oilseeds': {'yield_optimal': 1.2, 'water_mm': 600, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Sugarcane': {'yield_optimal': 70.0, 'water_mm': 2000, 'N_kg': 250, 'P_kg': 100, 'K_kg': 150},
    'Sweet potato': {'yield_optimal': 15.0, 'water_mm': 800, 'N_kg': 80, 'P_kg': 40, 'K_kg': 100},
    'Arhar/Tur': {'yield_optimal': 1.2, 'water_mm': 500, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Bajra': {'yield_optimal': 2.0, 'water_mm': 400, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Castor seed': {'yield_optimal': 1.5, 'water_mm': 500, 'N_kg': 60, 'P_kg': 40, 'K_kg': 40},
    'Coriander': {'yield_optimal': 1.0, 'water_mm': 500, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Cotton(lint)': {'yield_optimal': 2.5, 'water_mm': 800, 'N_kg': 120, 'P_kg': 60, 'K_kg': 60},
    'Gram': {'yield_optimal': 1.5, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Groundnut': {'yield_optimal': 2.0, 'water_mm': 600, 'N_kg': 25, 'P_kg': 50, 'K_kg': 30},
    'Horse-gram': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Jowar': {'yield_optimal': 2.0, 'water_mm': 400, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Linseed': {'yield_optimal': 1.0, 'water_mm': 500, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Maize': {'yield_optimal': 4.0, 'water_mm': 600, 'N_kg': 120, 'P_kg': 60, 'K_kg': 60},
    'Mesta': {'yield_optimal': 2.0, 'water_mm': 800, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Moong(Green Gram)': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Niger seed': {'yield_optimal': 0.8, 'water_mm': 500, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Onion': {'yield_optimal': 20.0, 'water_mm': 600, 'N_kg': 100, 'P_kg': 50, 'K_kg': 80},
    'Other Rabi pulses': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Potato': {'yield_optimal': 25.0, 'water_mm': 700, 'N_kg': 150, 'P_kg': 80, 'K_kg': 150},
    'Ragi': {'yield_optimal': 2.0, 'water_mm': 400, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Rapeseed&Mustard': {'yield_optimal': 1.5, 'water_mm': 500, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Safflower': {'yield_optimal': 1.0, 'water_mm': 500, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Sesamum': {'yield_optimal': 0.8, 'water_mm': 400, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Small millets': {'yield_optimal': 1.5, 'water_mm': 400, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Soyabean': {'yield_optimal': 2.5, 'water_mm': 600, 'N_kg': 40, 'P_kg': 60, 'K_kg': 40},
    'Sunflower': {'yield_optimal': 2.0, 'water_mm': 600, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Tapioca': {'yield_optimal': 25.0, 'water_mm': 1000, 'N_kg': 100, 'P_kg': 50, 'K_kg': 100},
    'Tobacco': {'yield_optimal': 2.0, 'water_mm': 600, 'N_kg': 80, 'P_kg': 40, 'K_kg': 60},
    'Turmeric': {'yield_optimal': 20.0, 'water_mm': 1300, 'N_kg': 120, 'P_kg': 60, 'K_kg': 100},
    'Urad': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Wheat': {'yield_optimal': 3.5, 'water_mm': 500, 'N_kg': 120, 'P_kg': 60, 'K_kg': 40},
    'Oilseeds total': {'yield_optimal': 1.5, 'water_mm': 600, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Jute': {'yield_optimal': 2.5, 'water_mm': 1000, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Masoor': {'yield_optimal': 1.2, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Peas & beans (Pulses)': {'yield_optimal': 1.5, 'water_mm': 500, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Barley': {'yield_optimal': 3.0, 'water_mm': 400, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Garlic': {'yield_optimal': 12.0, 'water_mm': 600, 'N_kg': 100, 'P_kg': 50, 'K_kg': 80},
    'Khesari': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Sannhamp': {'yield_optimal': 1.5, 'water_mm': 800, 'N_kg': 60, 'P_kg': 30, 'K_kg': 40},
    'Guar seed': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Moth': {'yield_optimal': 1.0, 'water_mm': 400, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Cardamom': {'yield_optimal': 0.3, 'water_mm': 1200, 'N_kg': 100, 'P_kg': 50, 'K_kg': 100},
    'Other Cereals': {'yield_optimal': 2.0, 'water_mm': 500, 'N_kg': 80, 'P_kg': 40, 'K_kg': 40},
    'Cowpea(Lobia)': {'yield_optimal': 1.2, 'water_mm': 500, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
    'Dry Ginger': {'yield_optimal': 5.0, 'water_mm': 1300, 'N_kg': 120, 'P_kg': 60, 'K_kg': 100},
    'Other Summer Pulses': {'yield_optimal': 1.0, 'water_mm': 500, 'N_kg': 20, 'P_kg': 40, 'K_kg': 20},
}

def crop_analysis(df, inputs):
    district = inputs['district'].upper()
    crop = inputs['crop']
    farm_size = inputs['farm_size_ha']
    season = inputs['season']
    irrigation = inputs['irrigation']
    prev_crop = inputs['previous_crop']

    crop_data = df[(df['District'] == district) & (df['Crop'] == crop)].copy()

    if crop_data.empty:
        return {"error": f"No data available for district '{district}' and crop '{crop}'."}

    season_yield = crop_data.groupby('Season')['Yield'].mean()
    optimal_season = season_yield.idxmax() if not season_yield.empty else season
    planting_time = f"Plant {crop} in {optimal_season} (avg yield: {season_yield.max():.2f} tonnes/ha)."

    avg_yield = crop_data['Yield'].mean()
    norms = crop_norms.get(crop, {'yield_optimal': 3.0, 'water_mm': 1000, 'N_kg': 100, 'P_kg': 40, 'K_kg': 60})
    yield_gap = norms['yield_optimal'] - avg_yield
    n_adj = norms['N_kg'] * (1 + yield_gap / norms['yield_optimal']) if yield_gap > 0 else norms['N_kg']
    p_adj = norms['P_kg']
    k_adj = norms['K_kg']
    if 'pulses' in prev_crop.lower():
        n_adj *= 0.8
    fertilizer = f"Apply {n_adj:.0f} kg Nitrogen, {p_adj:.0f} kg Phosphrous, {k_adj:.0f} kg Potassium per ha."

    water_need = norms['water_mm'] * farm_size
    water_status = "likely met" if season == 'Kharif' and irrigation == 'Yes' else "may need more irrigation"
    water = f"Needs {norms['water_mm']} mm/season ({water_need} mm for {farm_size} ha); {water_status}."

    yield_drop = crop_data['Yield'].max() - crop_data['Yield'].min()
    risk_level = 'Low' if yield_drop < 1.0 else ('Medium' if season == 'Kharif' else 'High')
    rotation_tip = "Maintain rotation" if crop == prev_crop else "Rotation reduces risk"
    pest_risk = f"{risk_level} risk (yield variation: {yield_drop:.2f}); {rotation_tip}."

    return {
        'Optimal Planting Time': planting_time,
        'Fertilizer Recommendation': fertilizer,
        'Water Requirement': water,
        'Pest & Disease Risk': pest_risk
    }

# --- Price Prediction Section (from price-prediction/server.py) ---
# Load CatBoost model for price prediction
price_model = CatBoostRegressor()
price_model.load_model('catboost_best.cbm')

# Load price dataset
price_df = pd.read_csv('large_crop_price_dataset.csv')

# Print columns for debugging
logger.info("Price dataset columns: %s", price_df.columns.tolist())

# Print unique crops and states for debugging
logger.info("Unique crops: %s", sorted(price_df['Commodity (Crop Name)'].unique().tolist()))
logger.info("Unique states: %s", sorted(price_df['State'].unique().tolist()))

# Extract Month and Year from Date
if 'Date' in price_df.columns:
    price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
    if 'Month' not in price_df.columns:
        price_df['Month'] = price_df['Date'].dt.month
    if 'Year' not in price_df.columns:
        price_df['Year'] = price_df['Date'].dt.year

# Compute Month_sin and Month_cos
if 'Month_sin' not in price_df.columns:
    price_df['Month_sin'] = np.sin(2 * np.pi * price_df['Month'] / 12)
if 'Month_cos' not in price_df.columns:
    price_df['Month_cos'] = np.cos(2 * np.pi * price_df['Month'] / 12)

# Recreate label encoders
price_label_encoders = {
    'State': LabelEncoder().fit(price_df['State']),
    'District': LabelEncoder().fit(price_df['District']),
    'Commodity (Crop Name)': LabelEncoder().fit(price_df['Commodity (Crop Name)']),
    'Variety': LabelEncoder().fit(price_df['Variety']),
    'Grade': LabelEncoder().fit(price_df['Grade'])
}

# Encode categorical columns in price_df
for col in ['State', 'District', 'Commodity (Crop Name)', 'Variety', 'Grade']:
    price_df[col] = price_label_encoders[col].transform(price_df[col])

# Compute interaction and squared features for the dataset
price_df['Rainfall_Temp_Interaction'] = price_df['Rainfall (mm)'] * price_df['Temperature (°C)']
price_df['Supply_Demand_Ratio'] = price_df['Market Supply (quintals)'] / (price_df['Market Demand (quintals)'] + 1e-5)
price_df['MSP_WPI_Ratio'] = price_df['Government MSP (₹)'] / (price_df['WPI'] + 1e-5)
price_df['Transport_Fertilizer_Cost'] = price_df['Transportation Cost (₹/quintal)'] * price_df['Fertilizer Cost (₹/acre)']
for feature in ['Rainfall (mm)', 'Temperature (°C)', 'Soil Moisture (%)', 'WPI', 'Government MSP (₹)']:
    price_df[f'{feature}_squared'] = price_df[feature] ** 2

# Define features and numerical columns for price prediction
price_features = ['State', 'District', 'Commodity (Crop Name)', 'Variety', 'Grade', 'Month', 'Year', 'Month_sin', 'Month_cos', 'Rainfall (mm)', 'Temperature (°C)', 'Soil Moisture (%)', 'WPI', 'Government MSP (₹)', 'Market Demand (quintals)', 'Stock Availability (quintals)', 'Market Supply (quintals)', 'Transportation Cost (₹/quintal)', 'Fertilizer Cost (₹/acre)', 'Export Demand (%)', 'Rainfall_Temp_Interaction', 'Supply_Demand_Ratio', 'MSP_WPI_Ratio', 'Transport_Fertilizer_Cost', 'Rainfall (mm)_squared', 'Temperature (°C)_squared', 'Soil Moisture (%)_squared', 'WPI_squared', 'Government MSP (₹)_squared']
price_numerical_cols = [f for f in price_features if f not in ['State', 'District', 'Commodity (Crop Name)', 'Variety', 'Grade', 'Month', 'Year', 'Month_sin', 'Month_cos']]

# Recreate scaler for price prediction
price_scaler = StandardScaler()
price_scaler.fit(price_df[price_numerical_cols])

def generate_detailed_prediction(date_str, crop, state, district, field_size):
    try:
        logger.info("Received prediction request: date=%s, crop=%s, state=%s, district=%s, field_size=%s",
                    date_str, crop, state, district, field_size)

        # Parse date
        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
        day = date_obj.day
        month = date_obj.month
        year = date_obj.year

        # Validate inputs
        if state not in price_label_encoders['State'].classes_:
            raise ValueError(f"State '{state}' not found in dataset.")
        if district not in price_label_encoders['District'].classes_:
            raise ValueError(f"District '{district}' not found in dataset.")
        if crop not in price_label_encoders['Commodity (Crop Name)'].classes_:
            raise ValueError(f"Crop '{crop}' not found in dataset.")
        if field_size <= 0:
            raise ValueError("Field size must be greater than 0.")

        # Encode categorical variables
        state_encoded = price_label_encoders['State'].transform([state])[0]
        district_encoded = price_label_encoders['District'].transform([district])[0]
        crop_encoded = price_label_encoders['Commodity (Crop Name)'].transform([crop])[0]

        # Define state_label and crop_label
        state_label = state_encoded
        crop_label = crop_encoded

        # Filter relevant data for Variety and Grade selection
        relevant_data = price_df[(price_df['State'] == state_label) & (price_df['Commodity (Crop Name)'] == crop_label)]
        if relevant_data.empty:
            raise ValueError("No data available for the given state and crop combination.")
        variety_encoded = relevant_data['Variety'].mode()[0]
        grade_encoded = relevant_data['Grade'].mode()[0]

        # Filter relevant data again
        relevant_data = price_df[(price_df['State'] == state_label) & (price_df['Commodity (Crop Name)'] == crop_label)]
        if relevant_data.empty:
            raise ValueError("No data available for the given state and crop combination.")

        avg_yield_per_acre = 15
        estimated_quantity = field_size * avg_yield_per_acre

        input_data = pd.DataFrame(columns=price_features)
        input_data.loc[0, 'State'] = state_encoded
        input_data.loc[0, 'District'] = district_encoded
        input_data.loc[0, 'Commodity (Crop Name)'] = crop_encoded
        input_data.loc[0, 'Variety'] = variety_encoded
        input_data.loc[0, 'Grade'] = grade_encoded
        input_data.loc[0, 'Month'] = month
        input_data.loc[0, 'Year'] = year
        input_data.loc[0, 'Month_sin'] = np.sin(2 * np.pi * month / 12)
        input_data.loc[0, 'Month_cos'] = np.cos(2 * np.pi * month / 12)

        # Populate numerical features with median values
        for feature in price_features:
            if feature not in ['State', 'District', 'Commodity (Crop Name)', 'Variety', 'Grade', 'Month', 'Year', 'Month_sin', 'Month_cos', 'Rainfall_Temp_Interaction', 'Supply_Demand_Ratio', 'MSP_WPI_Ratio', 'Transport_Fertilizer_Cost', 'Rainfall (mm)_squared', 'Temperature (°C)_squared', 'Soil Moisture (%)_squared', 'WPI_squared', 'Government MSP (₹)_squared']:
                input_data.loc[0, feature] = relevant_data[feature].median()

        # Compute interaction and squared features
        input_data['Rainfall_Temp_Interaction'] = input_data['Rainfall (mm)'] * input_data['Temperature (°C)']
        input_data['Supply_Demand_Ratio'] = input_data['Market Supply (quintals)'] / (input_data['Market Demand (quintals)'] + 1e-5)
        input_data['MSP_WPI_Ratio'] = input_data['Government MSP (₹)'] / (input_data['WPI'] + 1e-5)
        input_data['Transport_Fertilizer_Cost'] = input_data['Transportation Cost (₹/quintal)'] * input_data['Fertilizer Cost (₹/acre)']
        for feature in ['Rainfall (mm)', 'Temperature (°C)', 'Soil Moisture (%)', 'WPI', 'Government MSP (₹)']:
            input_data[f'{feature}_squared'] = input_data[feature] ** 2

        input_data = input_data.astype(float)
        input_data[price_numerical_cols] = price_scaler.transform(input_data[price_numerical_cols])

        predicted_price = price_model.predict(input_data)[0]

        predictions = []
        for _ in range(200):
            sample_data = input_data.copy()
            for feature in price_numerical_cols:
                sample_data[feature] += np.random.normal(0, 0.05 * relevant_data[feature].std())
            predictions.append(price_model.predict(sample_data)[0])
        ci_lower = np.percentile(predictions, 2.5)
        ci_upper = np.percentile(predictions, 97.5)

        min_price = relevant_data['Min Price (₹)'].min()
        max_price = relevant_data['Max Price (₹)'].max()

        district_prices = relevant_data.groupby('District')['Modal Price (₹)'].mean().sort_values(ascending=False)
        best_district_encoded = district_prices.index[0]
        best_district = price_label_encoders['District'].inverse_transform([best_district_encoded])[0]
        best_price = district_prices.iloc[0]

        future_month = min(month + 1, 12)
        input_data_future = input_data.copy()
        input_data_future['Month'] = future_month
        input_data_future['Month_sin'] = np.sin(2 * np.pi * future_month / 12)
        input_data_future['Month_cos'] = np.cos(2 * np.pi * future_month / 12)
        future_price = price_model.predict(input_data_future)[0]
        price_volatility = relevant_data['Modal Price (₹)'].std()
        storage_advice = "Sell now" if predicted_price >= future_price or price_volatility > predicted_price * 0.1 else "Hold for next month"

        avg_rainfall = relevant_data['Rainfall (mm)'].mean()
        avg_temp = relevant_data['Temperature (°C)'].mean()
        supply_demand_ratio = relevant_data['Market Supply (quintals)'].mean() / (relevant_data['Market Demand (quintals)'].mean() + 1e-5)
        export_demand = relevant_data['Export Demand (%)'].mean()
        if 50 <= avg_rainfall <= 150 and 20 <= avg_temp <= 30 and 0.8 <= supply_demand_ratio <= 1.2:
            market_trend = "Price stable due to balanced weather and supply-demand"
        elif supply_demand_ratio > 1.2 or avg_rainfall > 150 or export_demand < 10:
            market_trend = "Price may decrease due to oversupply, excess rain, or low export demand"
        else:
            market_trend = "Price may increase due to high demand, favorable weather, or strong export demand"

        expected_revenue = predicted_price * estimated_quantity

        return {
            "price_range": f"₹{min_price:.2f} - ₹{max_price:.2f} per quintal",
            "predicted_price": f"₹{predicted_price:.2f} per quintal (95% CI: ₹{ci_lower:.2f} - ₹{ci_upper:.2f})",
            "best_district": f"{best_district} (Higher price: ₹{best_price:.2f} per quintal)",
            "storage_advice": storage_advice,
            "market_trend": market_trend,
            "expected_revenue": f"₹{expected_revenue:.2f} for {estimated_quantity:.0f} quintals (estimated from {field_size} acres)"
        }

    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise ValueError(f"Prediction failed: {str(e)}")

# --- Irrigation Prediction Section (from irrigation/server.py) ---
# Define the path to the saved CatBoost model for irrigation
irrigation_catboost_path = os.path.abspath("catboost_model.cbm")

# Load the CatBoost model for irrigation
irrigation_model = CatBoostRegressor()
irrigation_model.load_model(irrigation_catboost_path)
print("Irrigation CatBoost model loaded successfully from", irrigation_catboost_path)

# Hardcoded preprocessing objects for irrigation
irrigation_label_encoders = {
    'Crop_Type': {'Wheat': 0, 'Rice': 1, 'Maize': 2},
    'Irrigation_Type': {'Drip': 0, 'Flood': 1, 'Sprinkler': 2},
    'Season': {'Kharif': 0, 'Rabi': 1, 'Summer': 2}
}

# Feature and target definitions for irrigation
irrigation_features = ['Crop_Type', 'Farm_Area(acres)', 'Irrigation_Type', 'Season', 'Motor_Capacity(HP)', 
                      'Pump_Flow_Rate(cubic meters/hour)', 'Irrigation_Depth_Applied(mm)', 'Irrigation_System_Efficiency(%)']
irrigation_numerical_cols = ['Farm_Area(acres)', 'Motor_Capacity(HP)', 'Pump_Flow_Rate(cubic meters/hour)', 
                            'Irrigation_Depth_Applied(mm)', 'Irrigation_System_Efficiency(%)']
irrigation_targets = ['Irrigation_Time(hours)', 'Total_Water_Needed(cubic meters)', 'Energy_Consumption(kWh)', 'Irrigation_System_Efficiency(%)']

# Dummy scalers for irrigation
class DummyScaler:
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X

irrigation_input_scaler = DummyScaler()
irrigation_target_scalers = {target: DummyScaler() for target in irrigation_targets}

# Hardcoded depth and efficiency maps for irrigation
irrigation_depth_map = {
    'Wheat': 5, 'Rice': 10, 'Maize': 6, 'Crop4': 7, 'Crop5': 5, 'Crop6': 8, 'Crop7': 4, 'Crop8': 5, 'Crop9': 4, 'Crop10': 3,
    'Crop11': 4, 'Crop12': 3, 'Crop13': 4, 'Crop14': 3, 'Crop15': 3, 'Crop16': 4, 'Crop17': 4, 'Crop18': 3, 'Crop19': 4, 'Crop20': 3,
    'Crop21': 4, 'Crop22': 6, 'Crop23': 4, 'Crop24': 5, 'Crop25': 6, 'Crop26': 4, 'Crop27': 4, 'Crop28': 4, 'Crop29': 3, 'Crop30': 4,
    'Crop31': 3, 'Crop32': 4, 'Crop33': 5, 'Crop34': 4, 'Crop35': 5, 'Crop36': 4, 'Crop37': 6, 'Crop38': 8, 'Crop39': 7, 'Crop40': 5,
    'Crop41': 6, 'Crop42': 5, 'Crop43': 4, 'Crop44': 6, 'Crop45': 8, 'Crop46': 9, 'Crop47': 7, 'Crop48': 5, 'Crop49': 6, 'Crop50': 7
}
irrigation_efficiency_map = {
    'Drip': 90, 'Flood': 80, 'Sprinkler': 50, 'Type4': 40, 'Type5': 60, 'Type6': 55, 'Type7': 60, 'Type8': 65, 'Type9': 85, 'Type10': 75,
    'Type11': 88, 'Type12': 50, 'Type13': 87, 'Type14': 45, 'Type15': 78, 'Type16': 76, 'Type17': 70, 'Type18': 82, 'Type19': 65, 'Type20': 68,
    'Type21': 50, 'Type22': 85, 'Type23': 45, 'Type24': 75, 'Type25': 70, 'Type26': 60, 'Type27': 55, 'Type28': 60, 'Type29': 65, 'Type30': 70
}

# --- Endpoints ---

# Crop Analysis Endpoints
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        result = crop_analysis(crop_df, data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/crops', methods=['GET'])
def get_crops():
    try:
        district = request.args.get('district', '').upper()
        if not district:
            return jsonify({'error': 'District is required'}), 400
        crops = crop_df[crop_df['District'] == district]['Crop'].unique().tolist()
        if not crops:
            return jsonify({'error': f'No crops found for district {district}'}), 404
        return jsonify({'crops': crops}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/districts', methods=['GET'])
def get_districts():
    try:
        districts = crop_df['District'].unique().tolist()
        if not districts:
            return jsonify({'error': 'No districts found in the data'}), 404
        return jsonify({'districts': districts}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Price Prediction Endpoints
@app.route('/options', methods=['GET'])
def get_price_options():
    try:
        # Get unique states and crops
        states = list(price_label_encoders['State'].classes_)
        crops = list(price_label_encoders['Commodity (Crop Name)'].classes_)

        # Create a state-district mapping
        state_districts = {}
        for state in states:
            state_encoded = price_label_encoders['State'].transform([state])[0]
            # Filter districts for this state
            districts_encoded = price_df[price_df['State'] == state_encoded]['District'].unique()
            # Decode district names
            districts = price_label_encoders['District'].inverse_transform(districts_encoded)
            state_districts[state] = sorted(list(districts))

        options = {
            "states": states,
            "crops": crops,
            "stateDistricts": state_districts
        }
        return jsonify(options), 200
    except Exception as e:
        logger.error("Error in get_price_options: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/crop-price-prediction', methods=['POST'])
def predict_price():
    try:
        input_data = request.get_json()
        result = generate_detailed_prediction(
            input_data['date_str'], input_data['crop'], input_data['state'],
            input_data['district'], input_data['field_size']
        )
        return jsonify(result), 200
    except ValueError as e:
        logger.error("Prediction endpoint error: %s", str(e))
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error("Unexpected error in predict endpoint: %s", str(e))
        return jsonify({'error': "Internal server error"}), 500

# Irrigation Prediction Endpoint
@app.route('/irrigation-prediction', methods=['POST'])
def predict_irrigation():
    try:
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        for col in ['Crop_Type', 'Irrigation_Type', 'Season']:
            if col not in input_data or input_data[col] not in irrigation_label_encoders[col]:
                raise ValueError(f"Invalid value for {col}: {input_data.get(col)}")
            input_df[col] = input_df[col].map(irrigation_label_encoders[col])
        
        # Derive additional features
        input_df['Pump_Flow_Rate(cubic meters/hour)'] = input_df['Motor_Capacity(HP)'] * 7.5
        input_df['Irrigation_Depth_Applied(mm)'] = input_df['Crop_Type'].map(lambda x: irrigation_depth_map[list(irrigation_label_encoders['Crop_Type'].keys())[x]])
        input_df['Irrigation_System_Efficiency(%)'] = input_df['Irrigation_Type'].map(lambda x: irrigation_efficiency_map[list(irrigation_label_encoders['Irrigation_Type'].keys())[x]])
        
        # Scale input features
        input_df[irrigation_numerical_cols] = irrigation_input_scaler.transform(input_df[irrigation_numerical_cols])
        input_df = input_df[irrigation_features]
        
        # Predict and debug raw output
        y_pred_scaled = irrigation_model.predict(input_df)
        print("Raw scaled predictions:", y_pred_scaled)
        
        predictions = {}
        for i, target in enumerate(irrigation_targets):
            pred_unscaled = irrigation_target_scalers[target].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).ravel()[0]
            predictions[target] = max(pred_unscaled, 0)
        
        # Manual calculations for adjustment
        manual_water = (input_data['Farm_Area(acres)'] * input_df['Irrigation_Depth_Applied(mm)'].iloc[0] * 4.04686) / (input_df['Irrigation_System_Efficiency(%)'].iloc[0] / 100)
        manual_time = manual_water / (input_data['Motor_Capacity(HP)'] * 7.5)
        manual_energy = input_data['Motor_Capacity(HP)'] * 0.746 * manual_time
        
        # Adjust predictions to align with manual calculations
        predictions['Total_Water_Needed(cubic meters)'] = min(max(predictions['Total_Water_Needed(cubic meters)'], 0), manual_water)
        predictions['Irrigation_Time(hours)'] = max(predictions['Irrigation_Time(hours)'], manual_time * 0.9)
        predictions['Energy_Consumption(kWh)'] = max(predictions['Energy_Consumption(kWh)'], manual_energy * 0.9)
        
        return jsonify(predictions), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)