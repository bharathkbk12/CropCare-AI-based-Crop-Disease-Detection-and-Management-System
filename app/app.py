from flask import Flask, render_template, request, redirect, jsonify
from markupsafe import Markup
import os
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import base64
import re

# Try to import heavy ML dependencies; allow app to run without them
try:
    import torch
    from torchvision import transforms
    from PIL import Image
    from utils.model import ResNet9
    ML_AVAILABLE = True
except Exception:
    torch = None
    transforms = None
    Image = None
    ResNet9 = None
    ML_AVAILABLE = False

# Register HEIC/HEIF support if available (for iPhone images)
try:
    from pillow_heif import register_heif_opener  # type: ignore
    register_heif_opener()
    print("HEIF support enabled for PIL.")
except Exception:
    pass

# Load environment variables from the .env file
load_dotenv()

# =============================== LOADING THE TRAINED MODELS =====================================

# Classes for plant disease classification
disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the disease classification model if ML deps are available
disease_model = None
if ML_AVAILABLE:
    # Use ResNet9 model (EfficientNet model file is corrupted - only 2KB)
    resnet_model_path = os.path.join(base_dir, 'models', 'plant_disease_model.pth')
    
    if os.path.exists(resnet_model_path):
        try:
            print("Loading ResNet9 disease model...")
            disease_model = ResNet9(3, len(disease_classes))
            disease_model.load_state_dict(torch.load(resnet_model_path, map_location=torch.device('cpu'), weights_only=False))
            disease_model.eval()
            print("✅ Loaded ResNet9 disease model successfully")
        except Exception as e:
            print(f"❌ Failed to load ResNet9 model: {e}")
            disease_model = None
    else:
        print(f"❌ ResNet9 model file not found: {resnet_model_path}")
        disease_model = None

# Load the crop recommendation model if file exists
crop_recommendation_model = None
crop_recommendation_model_path = os.path.join(base_dir, 'models', 'RandomForest.pkl')
if os.path.exists(crop_recommendation_model_path):
    try:
        crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))
    except Exception:
        crop_recommendation_model = None

# Lightweight rules-based fallback to avoid failures when the sklearn pickled
# model cannot be loaded or used due to version mismatch. This ensures the
# crop recommendation feature keeps working in a degraded but helpful mode.
def simple_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall):
    """Return a simple heuristic crop recommendation when ML model is unavailable.

    The rules below are basic and intended as a graceful fallback so the
    application remains useful even if the pickled sklearn model cannot be
    loaded due to version incompatibilities.
    """
    try:
        # Normalize inputs to safe ranges
        ph = max(3.5, min(9.0, float(ph)))
        rainfall = max(0.0, float(rainfall))
        temperature = float(temperature)
        humidity = float(humidity)
    except Exception:
        # If parsing fails, provide a sensible default
        return "rice"

    # Normalize macronutrients
    N = int(N); P = int(P); K = int(K)

    # More sophisticated rules that consider temperature and humidity differences
    
    # High humidity regions (tropical/subtropical) - PRIORITY 1
    if humidity >= 75:
        if temperature >= 25:
            if rainfall >= 150:
                return "rice"  # Tropical rice
            else:
                return "maize"  # Tropical maize
        else:
            return "rice"  # Cool humid = rice
    
    # Hot and dry regions - PRIORITY 2
    elif temperature >= 30 and humidity < 50:
        if rainfall >= 100:
            return "cotton"  # Hot dry with some rain = cotton
        else:
            return "sorghum"  # Hot dry = drought-tolerant sorghum
    
    # Moderate temperature regions - PRIORITY 3
    elif 20 <= temperature < 30:
        if humidity >= 60:
            if rainfall >= 120:
                return "maize"  # Moderate temp, humid = maize
            else:
                return "wheat"  # Moderate temp, humid, low rain = wheat
        else:
            if rainfall >= 100:
                return "wheat"  # Moderate temp, dry, good rain = wheat
            else:
                return "chickpea"  # Moderate temp, dry, low rain = chickpea
    
    # Cool regions - PRIORITY 4
    elif temperature < 20:
        if humidity >= 60:
            return "wheat"  # Cool humid = wheat
        else:
            if rainfall >= 80:
                return "wheat"  # Cool dry, good rain = wheat
            else:
                return "chickpea"  # Cool dry, low rain = chickpea
    
    # Special cases based on soil conditions - PRIORITY 5
    # Low pH regions
    if ph < 6.0 and N >= 30:
        return "soybean"  # Acidic soil = soybean
    
    # Low macronutrients overall
    if K < 30 and N < 30 and P < 30:
        return "pigeonpeas"  # Poor soil = pigeonpeas
    
    # High K and warm temps
    if K >= 60 and temperature >= 28 and rainfall >= 60:
        return "cotton"  # Rich soil, warm = cotton
    
    # Default fallback
    return "sorghum"

# ===============================================================================================

# Custom functions for fetching Google prediction and weather details

def fetch_google_prediction(query):
    """
    Fetch Google prediction based on a search query.
    :param query: The disease or crop-related search query
    :return: Google's prediction (string)
    """
    google_url = f"https://www.google.com/search?q={query}&ie=UTF-8"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(google_url, headers=headers)

    # Parse the search results to extract prediction (adjust as needed based on Google search result structure)
    soup = BeautifulSoup(response.text, 'html.parser')
    prediction = soup.find('div', class_='BNeawe iBp4i AP7Wnd')  # Update the class based on actual result
    return prediction.text if prediction else "No prediction found"

# ===============================================================================================

def weather_fetch(city_name: str):
    """Fetch weather for a city using an external API if configured, otherwise use location-based defaults.

    Returns: (temperature_c, humidity_pct, condition_text)
    """
    # Try to get API key from environment or config
    api_key = os.getenv("WEATHER_API_KEY") or getattr(config, 'weather_api_key', None)
    api_url = os.getenv("WEATHER_API_URL", "https://api.weatherapi.com/v1/current.json")

    if api_key:
        try:
            params = {"key": api_key, "q": city_name or "London", "aqi": "no"}
            resp = requests.get(api_url, params=params, timeout=10)
            if resp.ok:
                data = resp.json()
                current = data.get("current", {})
                condition = current.get("condition", {})
                temperature = float(current.get("temp_c", 25))
                humidity = float(current.get("humidity", 60))
                condition_text = str(condition.get("text", "Partly cloudy"))
                print(f"Weather fetched for {city_name}: {temperature}°C, {humidity}% humidity")
                return temperature, humidity, condition_text
            else:
                print(f"Weather API returned status {resp.status_code} for city {city_name}")
        except Exception as e:
            print(f"weather_fetch: API error for city {city_name}: {e}")

    # Location-based defaults instead of neutral defaults
    city_lower = city_name.lower() if city_name else "london"
    
    # Provide more realistic location-based defaults
    location_defaults = {
        "london": (15.0, 75.0, "Cloudy"),
        "mumbai": (30.0, 80.0, "Humid"),
        "delhi": (35.0, 45.0, "Hot"),
        "bangalore": (25.0, 70.0, "Pleasant"),
        "chennai": (32.0, 85.0, "Hot and Humid"),
        "kolkata": (28.0, 80.0, "Humid"),
        "hyderabad": (30.0, 65.0, "Warm"),
        "pune": (28.0, 60.0, "Pleasant"),
        "ahmedabad": (32.0, 50.0, "Hot"),
        "jaipur": (30.0, 40.0, "Dry"),
        "new york": (20.0, 65.0, "Cool"),
        "tokyo": (18.0, 70.0, "Mild"),
        "paris": (15.0, 70.0, "Cool"),
        "sydney": (22.0, 60.0, "Mild"),
        "singapore": (28.0, 85.0, "Hot and Humid")
    }
    
    # Find matching city or use closest match
    for city_key, defaults in location_defaults.items():
        if city_key in city_lower:
            print(f"Using location-based defaults for {city_name}: {defaults[0]}°C, {defaults[1]}% humidity")
            return defaults
    
    # Default fallback with some variation based on city name
    base_temp = 25.0
    base_humidity = 60.0
    
    # Add some variation based on city name hash
    city_hash = hash(city_name) if city_name else 0
    temp_variation = (city_hash % 20) - 10  # -10 to +10 degree variation
    humidity_variation = (city_hash % 30) - 15  # -15 to +15% variation
    
    final_temp = max(5.0, min(45.0, base_temp + temp_variation))
    final_humidity = max(20.0, min(95.0, base_humidity + humidity_variation))
    
    print(f"Using calculated defaults for {city_name}: {final_temp}°C, {final_humidity}% humidity")
    return final_temp, final_humidity, "Partly cloudy"

def predict_image(img, model=disease_model):
    """
    Transform image to tensor and predict disease label.
    :param img: Image file
    :param model: Trained PyTorch model
    :return: Prediction (string)
    """
    if not ML_AVAILABLE or model is None:
        raise RuntimeError("ML components are not available on this server.")

    # ResNet9 preprocessing - use larger input size to avoid pooling issues
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),  # Use 256x256 instead of 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(io.BytesIO(img)).convert('RGB')
        img_t = base_transform(image)
        img_u = torch.unsqueeze(img_t, 0)

        # Test-time augmentation: horizontal flip averaging
        with torch.no_grad():
            logits = model(img_u)
            logits_flipped = model(torch.flip(img_u, dims=[3]))
            logits_avg = (logits + logits_flipped) / 2.0
            probs = torch.softmax(logits_avg, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())

        # Safe mapping to class label
        if 0 <= pred_idx < len(disease_classes):
            return disease_classes[pred_idx]
        return "Unknown"
        
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise RuntimeError(f"Could not process image: {e}")

# ===============================================================================================

# ------------------------------------ FLASK APP -------------------------------------------------

app = Flask(__name__)
# Allow up to 10MB uploads to accommodate high-res photos
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Render home page
@app.route('/')
def home():
    title = 'CropCare 🌾🌾- Home'
    return render_template('index.html', title=title)

# Render crop recommendation form page
@app.route('/crop-recommend')
def crop_recommend():
    title = 'CropCare - Crop Recommendation'
    return render_template('crop.html', title=title)

# Render fertilizer recommendation form page
@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'CropCare - Fertilizer Suggestion'
    return render_template('fertilizer.html', title=title)

# Render crop recommendation result page
@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'CropCare - Crop Recommendation'

    if request.method == 'POST':
        try:
            N = int(request.form.get('nitrogen', 0))
            P = int(request.form.get('phosphorous', 0))
            K = int(request.form.get('pottasium', 0))
            ph = float(request.form.get('ph', 7))
            rainfall = float(request.form.get('rainfall', 0))
        except ValueError:
            return render_template('try_again.html', title=title)

        city_raw = request.form.get("city", "") or ""
        city = city_raw.strip() or "London"

        temperature, humidity, condition = weather_fetch(city)

        # Use fallback rules for location-based recommendations
        final_prediction = simple_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall)
        return render_template('crop-result.html', prediction=final_prediction,
                               temperature=temperature, humidity=humidity, condition=condition, city=city,
                               N=N, P=P, K=K, ph=ph, rainfall=rainfall,
                               source="fallback", title=title)

        # Always use the weather data we fetched
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        if crop_recommendation_model is not None:
            try:
                my_prediction = crop_recommendation_model.predict(data)
                final_prediction = my_prediction[0]
                print(f"crop_prediction: using ML model prediction for {city} - {final_prediction}")
                return render_template('crop-result.html', prediction=final_prediction,
                                       temperature=temperature, humidity=humidity, condition=condition, city=city,
                                       N=N, P=P, K=K, ph=ph, rainfall=rainfall,
                                       source="ml", title=title)
            except Exception as e:
                print(f"crop_prediction: sklearn.predict failed: {e}")
                # Gracefully fall back if sklearn prediction fails at runtime
                final_prediction = simple_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall)
                print(f"crop_prediction: using fallback rules for {city} - {final_prediction}")
                return render_template('crop-result.html', prediction=final_prediction,
                                       temperature=temperature, humidity=humidity, condition=condition, city=city,
                                       N=N, P=P, K=K, ph=ph, rainfall=rainfall,
                                       source="fallback", title=title)
        else:
            # Use fallback rules
            final_prediction = simple_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall)
            print(f"crop_prediction: using fallback rules for {city} - {final_prediction}")
            return render_template('crop-result.html', prediction=final_prediction,
                                   temperature=temperature, humidity=humidity, condition=condition, city=city,
                                   N=N, P=P, K=K, ph=ph, rainfall=rainfall,
                                   source="fallback", title=title)

# Render fertilizer recommendation result page
@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'CropCare - Fertilizer Suggestion'
    crop_name = str(request.form.get('cropname', '')).strip()
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    df = pd.read_csv(os.path.join(base_dir, 'Data', 'fertilizer.csv'))
    # Normalize both sides to lowercase for case-insensitive match
    rows = df[df['Crop'].str.lower() == crop_name.lower()]
    if rows.empty:
        return render_template('try_again.html', title=title)

    nr = rows['N'].iloc[0]
    pr = rows['P'].iloc[0]
    kr = rows['K'].iloc[0]

    n, p, k = nr - N, pr - P, kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    key = f"{max_value}{'High' if eval(max_value.lower()) < 0 else 'low'}"
    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# Render disease prediction result page
@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'CropCare - Disease Detection'

    if request.method == 'POST':
        file = request.files.get('file')
        camera_data_url = request.form.get('camera_image')
        img_bytes = None

        # Prefer uploaded file if present
        if file and getattr(file, 'filename', ''):
            img_bytes = file.read()
        elif camera_data_url:
            # Expect a data URL like: data:image/png;base64,....
            try:
                match = re.match(r"^data:image/[^;]+;base64,(.+)$", camera_data_url)
                if match:
                    img_bytes = base64.b64decode(match.group(1))
            except Exception:
                img_bytes = None

        if img_bytes is None:
            return render_template('disease.html', title=title, error_message="Please upload or capture an image.")

        # Guard against missing model but don't crash the view
        if disease_model is None:
            return render_template('disease.html', title=title, error_message="Disease model is not available on this server.")

        # Predict with the model — if it fails, show the form instead of crashing
        try:
            model_prediction = predict_image(img_bytes)
        except Exception as e:
            print(f"disease_prediction: model error: {e}")
            return render_template('disease.html', title=title, error_message="Could not analyze the image. Try another image.")

        # Try to enrich with Google, but don't fail the page if scraping fails
        try:
            google_prediction = fetch_google_prediction(model_prediction)
        except Exception as e:
            print(f"disease_prediction: google fetch error: {e}")
            google_prediction = "No prediction found"

        # Weather info — resilient to API/network issues
        city = request.form.get("city", "London")
        try:
            temperature, humidity, condition = weather_fetch(city)
        except Exception as e:
            print(f"disease_prediction: weather fetch error: {e}")
            temperature, humidity, condition = 25.0, 60.0, "Partly cloudy"

        # Extract crop type from disease prediction (e.g., "Apple___Black_rot" -> "Apple")
        detected_crop = model_prediction.split('___')[0] if '___' in model_prediction else model_prediction
        
        # Map prediction and display results (fallback to raw label if missing)
        mapped = disease_dic.get(model_prediction, model_prediction)
        model_prediction = Markup(str(mapped))
        google_prediction = Markup(f"Google's Prediction: {google_prediction}")

        # Render the template with predictions and weather data
        return render_template('disease-result.html', 
                               model_prediction=model_prediction, 
                               google_prediction=google_prediction, 
                               detected_crop=detected_crop,
                               temperature=temperature, 
                               humidity=humidity, 
                               condition=condition, city=city, title=title)

    return render_template('disease.html', title=title)

# AI Search endpoint
@app.route('/ai_search', methods=['POST'])
def ai_search():
    """AI-powered search functionality for agricultural queries"""
    try:
        data = request.get_json()
        query = data.get('query', '').lower().strip()
        
        if not query:
            return jsonify({'answer': 'Please enter a search query.'})
        
        # Simple response based on query content
        if 'apple' in query and 'scab' in query:
            answer = "🦠 **Apple Scab Disease:**\nApple Scab is caused by the fungus Venturia inaequalis. Symptoms include dark spots on leaves and fruit. Control with fungicides and proper pruning. For accurate diagnosis, use our Disease Detection tool!"
        elif 'wheat' in query:
            answer = "🌾 **Wheat Information:**\nWheat is a cereal grain that requires moderate rainfall (400-600mm), temperature 15-25°C, and well-drained soil with pH 6.0-7.5. It's suitable for temperate climates."
        elif 'fertilizer' in query or 'nutrient' in query:
            answer = "🌱 **Fertilizer Information:**\nFor personalized fertilizer recommendations, use our Fertilizer Recommendation tool with your crop and soil data. Our system considers NPK levels, soil pH, and weather conditions."
        else:
            answer = "🤖 **AI Assistant:**\nI can help you with crop information, disease identification, and fertilizer advice. Try asking about specific crops, diseases, or fertilizers. For detailed analysis, use our specialized tools!"
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"AI search error: {e}")
        return jsonify({'answer': 'Sorry, I encountered an error. Please try again or use our specialized tools for detailed analysis.'})

# ===============================================================================================

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    host = os.getenv('HOST', '0.0.0.0')
    app.run(host=host, port=port, debug=False)
    