import torch  
import os
from PIL import Image
import torchvision.transforms as transforms

# Optional timm import with helpful error if missing
try:
    import timm
except Exception as exc:
    timm = None
    _timm_import_error = exc

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class CropDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(CropDiseaseModel, self).__init__()
        if timm is None:
            raise ImportError("timm is required for CropDiseaseModel but failed to import")
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Class labels (38 classes to match training)
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
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spo',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Load the model
model = CropDiseaseModel(num_classes=len(disease_classes))
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "efficientnet_b3_disease_model.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

state = torch.load(model_path, map_location=device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=False)
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        predicted_class = int(torch.argmax(logits, dim=1).item())
    
    if 0 <= predicted_class < len(disease_classes):
        return disease_classes[predicted_class]
    return "Unknown"
