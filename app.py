import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from preprocess import preprocess

# -----------------------------
# 1. YOUR EXACT CNN MODEL
# -----------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):  # change if your n_classes != 2
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((3, 3))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# 2. LOAD MODEL
# -----------------------------
device = torch.device("cpu")

model = SimpleCNN(num_classes=2)  # ⚠️ change if needed
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


# -----------------------------
# 3. STREAMLIT UI
# -----------------------------
st.title("🩺 Breast Cancer Detection System")
st.write("Upload a breast image and get prediction (Benign / Malignant)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# -----------------------------
# 4. PREDICTION FUNCTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocess
    image = preprocess(image)

    # prediction
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    # result
    st.markdown("---")
    st.subheader("Result")

    if prediction == 0:
        st.success("🟢 Benign")
    else:
        st.error("🔴 Malignant")