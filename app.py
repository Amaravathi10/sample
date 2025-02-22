from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

def t(text):
    try:
        return translator.translate(text, dest='te').text
    except Exception:
        return text

# Load datasets globally
df = pd.read_csv('data/Crop_recommendation.csv')
data = pd.read_csv('data/filled_combined_dataset.csv')
additional_data = pd.read_csv('data/new_dataset.csv')
seasonal_data = pd.read_csv('data/seasonaldata.csv')

# Feature 1: NPK -> Crop + Adjustments
avg_temp, avg_humidity, avg_ph, avg_rainfall = df[['temperature', 'humidity', 'ph', 'rainfall']].mean()
X_f1 = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_f1 = df['label']
X_train_f1, _, y_train_f1, _ = train_test_split(X_f1, y_f1, test_size=0.2, random_state=42)
rf_model_f1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_f1.fit(X_train_f1, y_train_f1)

# Feature 2: Crop -> NPK
le_f2 = LabelEncoder()
df['label_encoded'] = le_f2.fit_transform(df['label'].str.lower())
X_f2 = df[['label_encoded']]
y_f2 = df[['N', 'P', 'K']]
X_train_f2, _, y_train_f2, _ = train_test_split(X_f2, y_f2, test_size=0.2, random_state=42)
rf_model_f2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
rf_model_f2.fit(X_train_f2, y_train_f2)

# Feature 3 & 4: Cropping Strategies & Geo Recommendation
categorical_columns = ['Crop', 'CNext', 'CLast', 'CTransp', 'IrriType', 'IrriSource', 'Season', 'District']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
numerical_columns = ['expected_rainfall', 'temp_min', 'temp_max', 'humidity_min', 'humidity_max',
                     'wind_speed_min', 'wind_speed_max', 'N', 'P', 'K', 'temperature', 'humidity',
                     'ph', 'rainfall_required', 'ExpYield']
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())
features = ['District', 'expected_rainfall', 'temp_min', 'temp_max', 'humidity_min', 'humidity_max',
            'wind_speed_min', 'wind_speed_max', 'Season', 'N', 'P', 'K', 'temperature', 'humidity', 'ph']
X_f3 = data[features]
y_f3 = data['Crop']
X_train_f3, _, y_train_f3, _ = train_test_split(X_f3, y_f3, test_size=0.2, random_state=42)
model_f3 = GradientBoostingRegressor()
model_f3.fit(X_train_f3, y_train_f3)
model_f4 = GradientBoostingRegressor()
model_f4.fit(X_train_f3, y_train_f3)  # Same training data for simplicity

@app.route('/')
def home():
    return render_template('index.html')

# Feature 1: NPK -> Crop + Adjustments
@app.route('/feature1', methods=['GET', 'POST'])
def feature1():
    if request.method == 'POST':
        N, P, K = float(request.form['n']), float(request.form['p']), float(request.form['k'])
        input_data = np.array([[N, P, K, avg_temp, avg_humidity, avg_ph, avg_rainfall]])
        crop = rf_model_f1.predict(input_data)[0]
        similar_crops = find_similar_crops(N, P, K, crop)
        return jsonify({'crop': t(crop), 'similar_crops': similar_crops})
    return render_template('feature1.html')

def find_similar_crops(N, P, K, main_crop):
    crop_avg = df.groupby('label')[['N', 'P', 'K']].mean().reset_index()
    nbrs = NearestNeighbors(n_neighbors=6, metric='euclidean').fit(crop_avg[['N', 'P', 'K']])
    distances, indices = nbrs.kneighbors([[N, P, K]])
    similar = []
    for idx, dist in zip(indices[0], distances[0]):
        crop_data = crop_avg.iloc[idx]
        if crop_data['label'] != main_crop:
            similar.append({
                'crop': t(crop_data['label']),
                'N_adj': t(f"{crop_data['N'] - N:.2f}"),
                'P_adj': t(f"{crop_data['P'] - P:.2f}"),
                'K_adj': t(f"{crop_data['K'] - K:.2f}")
            })
    return similar[:5]

# Feature 2: Crop -> NPK
@app.route('/feature2', methods=['GET', 'POST'])
def feature2():
    crops = sorted(df['label'].unique())
    if request.method == 'POST':
        crop = request.form['crop'].lower()
        crop_encoded = le_f2.transform([crop])[0]
        npk_pred = rf_model_f2.predict([[crop_encoded]])[0]
        crop_data = df[df['label'].str.lower() == crop]
        return jsonify({
            'crop': t(crop),
            'N': {'predicted': npk_pred[0], 'min': crop_data['N'].min(), 'max': crop_data['N'].max()},
            'P': {'predicted': npk_pred[1], 'min': crop_data['P'].min(), 'max': crop_data['P'].max()},
            'K': {'predicted': npk_pred[2], 'min': crop_data['K'].min(), 'max': crop_data['K'].max()}
        })
    return render_template('feature2.html', crops=[t(c) for c in crops])

# Feature 3: Cropping Strategies
@app.route('/feature3', methods=['GET', 'POST'])
def feature3():
    if request.method == 'POST':
        district, area = request.form['district'], float(request.form['area'])
        recommendations = predict_crops_with_details(district, area)
        return jsonify({'recommendations': recommendations})
    return render_template('feature3.html')

def predict_crops_with_details(district, total_area):
    try:
        district_encoded = label_encoders['District'].transform([district])[0]
    except ValueError:
        return [{'Name': t('Error: District not found'), 'Area': 0}]
    avg_temp = data[data['District'] == district_encoded][['temp_min', 'temp_max']].mean().mean()
    avg_humidity = data[data['District'] == district_encoded][['humidity_min', 'humidity_max']].mean().mean()
    input_data = pd.DataFrame([{
        'District': district_encoded, 'expected_rainfall': data['expected_rainfall'].mean(),
        'temp_min': avg_temp, 'temp_max': avg_temp + 5, 'humidity_min': avg_humidity,
        'humidity_max': avg_humidity + 10, 'wind_speed_min': data['wind_speed_min'].mean(),
        'wind_speed_max': data['wind_speed_max'].mean(), 'Season': label_encoders['Season'].transform(['Kharif'])[0],
        'N': data['N'].mean(), 'P': data['P'].mean(), 'K': data['K'].mean(),
        'temperature': avg_temp, 'humidity': avg_humidity, 'ph': data['ph'].mean()
    }])
    crop_pred = model_f3.predict(input_data)[0]
    primary_crop = label_encoders['Crop'].inverse_transform([int(round(crop_pred))])[0]
    district_crops = additional_data[additional_data['District'] == district].sort_values('Yield (kg/ha)', ascending=False)
    recs = [{'Name': t(primary_crop), 'Area': total_area * 0.4}]
    remaining_area = total_area * 0.6 / min(len(district_crops), 3) if len(district_crops) > 0 else 0
    for _, crop in district_crops.head(3).iterrows():
        recs.append({'Name': t(crop['Crop']), 'Area': remaining_area, 'Yield': crop['Yield (kg/ha)']})
    return recs

# Feature 4: District + Season -> Crop Recommendation
@app.route('/feature4', methods=['GET', 'POST'])
def feature4():
    if request.method == 'POST':
        district, season = request.form['district'], request.form.get('season', '').strip()
        result = predict_crops(district, season)
        return jsonify(result)
    return render_template('feature4.html')

def predict_crops(district, season):
    if season:
        district_season_data = seasonal_data[
            (seasonal_data['District'].str.lower() == district.lower()) &
            (seasonal_data['Season'].str.lower() == season.lower())
        ]
        if not district_season_data.empty:
            return {'Recommendations': [{'Crop': t(row['Crop']), 'Variety': row['Variety']} for _, row in district_season_data.iterrows()]}
        return {'error': t(f"No data for {district} and {season}")}
    try:
        district_encoded = label_encoders['District'].transform([district])[0]
    except ValueError:
        return {'error': t(f"District {district} not found")}
    input_data = pd.DataFrame([{
        'District': district_encoded, 'expected_rainfall': data['expected_rainfall'].mean(),
        'temp_min': data['temp_min'].mean(), 'temp_max': data['temp_max'].mean(),
        'humidity_min': data['humidity_min'].mean(), 'humidity_max': data['humidity_max'].mean(),
        'wind_speed_min': data['wind_speed_min'].mean(), 'wind_speed_max': data['wind_speed_max'].mean(),
        'Season': label_encoders['Season'].transform(['Kharif'])[0], 'N': data['N'].mean(),
        'P': data['P'].mean(), 'K': data['K'].mean(), 'temperature': data['temperature'].mean(),
        'humidity': data['humidity'].mean(), 'ph': data['ph'].mean()
    }])
    crop_pred = model_f4.predict(input_data)[0]
    return {'Predicted Crop': t(label_encoders['Crop'].inverse_transform([int(round(crop_pred))])[0])}

if __name__ == '__main__':
    app.run(debug=True)
