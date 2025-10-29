import torch
import timm  # pyright: ignore[reportMissingImports]

# Define model class
class CropDiseaseModel(torch.nn.Module):
    def __init__(self, num_classes=38):  
        super(CropDiseaseModel, self).__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Create a model instance
model = CropDiseaseModel(num_classes=38)

# Define path to save the model
model_path = "C:/onedrive/Desktop/Crop/CropCare/models/efficientnet_b3_disease_model.pth"

# Save the trained model
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved at {model_path}")
