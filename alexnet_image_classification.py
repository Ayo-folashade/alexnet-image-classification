import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import AlexNet_Weights

# Load the pre-trained AlexNet model
alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)

# Set the model to evaluation mode
alexnet.eval()

# Preprocess the input image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load and preprocess the input image
image_path = 'test_images/dog.jpeg'
image = Image.open(image_path)
preprocessed_image = preprocess(image)
input_batch = preprocessed_image.unsqueeze(0)  # Add a batch dimension

# Forward pass through the network
output = alexnet(input_batch)

# Load the ImageNet class labels
with open('imagenet1000_clsidx_to_labels.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Get the predicted class index
_, predicted_idx = torch.max(output, 1)
predicted_label = classes[predicted_idx.item()]

# Print the predicted label
print('Predicted label:', predicted_label)
