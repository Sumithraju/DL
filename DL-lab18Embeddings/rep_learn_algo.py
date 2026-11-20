
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms
class TextRestoration(nn.Module):
    def __init__(self, max_features=100):
        super().__init__()
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')  # Initialize TF-IDF
    def fit_transform(self, texts):
        vectors = self.vectorizer.fit_transform(texts)  # Convert text into vectors
        return vectors.toarray()
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()  # Return vocabulary
class ImageRestoration(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')  # Select CPU/GPU
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load pretrained ResNet18
        self.model = nn.Sequential(*list(backbone.children())[:-1])  # Remove classification layer
        self.model.to(self.device)
        self.model.eval()  # Evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])
    def extract_features_from_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Load RGB image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Prepare input
        with torch.no_grad():
            features = self.model(image_tensor)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten features
        return features.cpu().numpy()[0]  # Return 512 features
    def extract_features_from_folder(self, folder_path):
        features = []
        image_names = []
        valid_ext = ('.jpg', '.jpeg', '.png')  # Supported extensions
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith(valid_ext):
                image_path = os.path.join(folder_path, file)  # Full image path
                if not os.path.isfile(image_path):
                    continue
                try:
                    feature_vector = self.extract_features_from_image(image_path)  # Extract embedding
                    features.append(feature_vector)
                    image_names.append(file)
                    print("Processed:", file)
                except Exception as e:
                    print("Error processing", file, ":", e)
        if len(features) == 0:
            raise ValueError("No valid images were processed.")
        return np.array(features), image_names  # Return feature matrix
class BiologicalRepresentation:
    def __init__(self, n_components=2):
        self.scaler = StandardScaler()  # Standardize gene values
        self.pca = PCA(n_components=n_components)  # Initialize PCA
    def fit_transform(self, gene_expression):
        scaled_data = self.scaler.fit_transform(gene_expression)  # Scale expression
        reduced_data = self.pca.fit_transform(scaled_data)  # Reduce dimensions
        return reduced_data
    def explained_variance(self):
        return self.pca.explained_variance_ratio_  # Variance explained
texts = [
    "Machine learning is useful in biology",
    "Deep learning is applied to medical images",
    "Bioinformatics uses machine learning",
    "Gene expression analysis is important in biology"
]
text_model = TextRestoration(max_features=100)  # Create text model
text_features = text_model.fit_transform(texts)  # Generate TF-IDF vectors
print("\nTEXT REPRESENTATION")
print("Shape:", text_features.shape)
print("Features:", text_model.get_feature_names())
print("TF-IDF vectors:\n", text_features)
image_folder = "images"  # Set image folder path
if os.path.isdir(image_folder):
    image_model = ImageRestoration()  # Initialize image encoder
    image_features, image_names = image_model.extract_features_from_folder(image_folder)
    print("\nIMAGE REPRESENTATION")
    print("Images:", image_names)
    print("Feature shape:", image_features.shape)
    if len(image_features) >= 3:
        pca_image = PCA(n_components=2)  # Initialize image PCA
        image_2d = pca_image.fit_transform(image_features)  # Reduce image features
        plt.figure(figsize=(8, 6))
        plt.scatter(image_2d[:, 0], image_2d[:, 1])  # Plot image embeddings
        for i, name in enumerate(image_names):
            plt.annotate(name, (image_2d[i, 0], image_2d[i, 1]))  # Label images
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Image Feature Representation")
        plt.tight_layout()
        plt.show()
else:
    print("\nImage folder not found. Skipping images.")
np.random.seed(42)  # Set random seed
gene_expression = np.random.poisson(lam=5, size=(20, 100))  # Generate synthetic gene counts
bio_model = BiologicalRepresentation(n_components=2)  # Initialize biological PCA
bio_features = bio_model.fit_transform(gene_expression)  # Reduce gene dimensions
print("\nBIOLOGICAL REPRESENTATION")
print("Original shape:", gene_expression.shape)
print("PCA shape:", bio_features.shape)
print("Explained variance:", bio_model.explained_variance())
plt.figure(figsize=(8, 6))
plt.scatter(bio_features[:, 0], bio_features[:, 1])  # Plot samples
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Gene Expression PCA")
plt.tight_layout()
plt.show()
