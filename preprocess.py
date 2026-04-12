from torchvision import transforms

data_transform = transforms.Compose([
    transforms.Grayscale(),        # important if model uses 1 channel
    transforms.Resize((28, 28)),   # MUST match training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess(image):
    image = data_transform(image)
    return image.unsqueeze(0)