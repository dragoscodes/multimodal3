import json
import boto3
from image_handling.padding import tensor_resize_with_padding, load_images_as_tensors, tensor_resize_without_padding
import torch
from tqdm import tqdm
from transformers import PretrainedConfig
import os
import torch.nn as nn
from dotenv import load_dotenv
from arch import LeMultiModal

# Load environment variables from .env file
load_dotenv()

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')

client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

#Download the images from s3
def download_image(bucket_name, image_path):
    client.download_file(bucket_name, image_path, 'tmp.jpg')

def download_data(bucket_name, image_path):
    client.download_file(bucket_name, image_path, 'data.json')

download_data("multimodal-ai-dataset", "sharegpt4v/sharegpt4v_instruct_gpt4-vision_cap100k.json")

def load_img(image_path):
    download_image("multimodal-ai-dataset", image_path)
    images = load_images(["tmp.jpg"])
    image = images[0]
    return image

def load_img_as_tensor(image_path):
    download_image("multimodal-ai-dataset", image_path)
    images = load_images_as_tensors(["tmp.jpg"])
    image = images[0]
    return image

with open('data.json') as f:
    data = json.load(f)

img_link.split("/")[-1]


config = PretrainedConfig
model = LeMultiModal(config)

num_epochs = 5  # Adjust as needed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(lm_head.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in tqdm(range(num_epochs)):
    for item in tqdm(data):  # Assuming your dataloader provides individual items
        image_path = item["image"]
        image = load_img(image_path)
        train_text = item['conversations'][0]['value']
        valid_text = item['conversations'][1]['value']

        model(image, train_text)

        lm_outputs = lm_outputs.logits.argmax(-1)[0]
        #Get embeddings for valid_text
        print('Model:' ,lm_outputs)
        print('Text: ', valid_text)

        loss = nn.CrossEntropyLoss()(lm_outputs, valid_text).mean()
        loss.backward()
        scheduler.step()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(lm_head.state_dict(), 'lm_head.pth')

    
with torch.no_grad():
    image = load_img_as_tensor("coco/train/000000000009.jpg")
    resized_img = tensor_resize_without_padding(image, 336)
    image_tensors = image_processor(resized_img, return_tensors="pt")["pixel_values"]
    batch_features = vision_model(image_tensors, output_hidden_states=True)
    image_features = batch_features.hidden_states[-1]
    print(image_features.shape)
    image_features = feature_select(image_features, "patch")

for epoch in range(num_epochs):
    for images, texts in dataloader:
        input_ids = self.tokenizer(texts, return_tensors="pt").input_ids.to(self.device)
        outputs = model(images, input_ids)  # Call the forward method
        loss = calculate_loss(outputs, labels)  # Define your loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()