import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

from preprocess import preprocess

# -----------------------------
# 1. DEFINE MODEL (MUST MATCH TRAINING)
# -----------------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# -----------------------------
# 2. LOAD MODEL
# -----------------------------
device = torch.device("cpu")

model = CNNModel()
model.load_state_dict(torch.load("modelBC_kaggle.pth", map_location=device))
model.to(device)
model.eval()


# -----------------------------
# 3. STREAMLIT UI
# -----------------------------
st.title("🩺 Breast Cancer Detection System")
st.write("Upload an ultrasound image to classify as Benign or Malignant")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])


# -----------------------------
# 4. PREDICTION
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    # preprocess
    image = preprocess(image)

    # prediction
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    # -----------------------------
    # 5. RESULT (CORRECT LABELS)
    # -----------------------------
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("🟢 Benign")
    else:
        st.error("🔴 Malignant")
