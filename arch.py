import torch
from transformers import PreTrainedModel
from loader.model_loader import load_vision_model, load_llm
from vision.projector import load_vision_projector
from vision.feature_select import feature_select
from vision.learned_encoding import load_learned_positional
from image_handling.padding import resize_with_padding, load_images
from image_handling.slice import split_image
from transformers import BitsAndBytesConfig
import math
import requests
from PIL import Image
from io import BytesIO

class LeMultiModalConfig:
    def __init__(self, 
                 max_len=8, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 vision_model_path="openai/clip-vit-large-patch14-336",
                 llm_model_path="SweatyCrayfish/llama-3-8b-quantized",
                 positional_encoding_type="sinusoidal",  # Or "learned", "none"
                 **kwargs):
        self.max_len = max_len
        self.device = device
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.positional_encoding_type = positional_encoding_type

class LeMultiModal(nn.Module):
    def __init__(self, config :LeMultiModalConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        self.max_len = config.max_len
        self.quantization_config = BitsAndBytesConfig(load_in_8bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.vision_model , self.image_processor = load_vision_model(config.vision_model_path, device = self.device )
        self.llm, self.tokenizer = load_llm(config.llm_model_path, device = self.device, quantization_config = self.quantization_config)
        self.vision_projector = load_vision_projector()
        self.llm_dim = self.llm.config.hidden_size
        self.vision_dim = self.vision_model.config.hidden_size
        self.learned_positional = load_learned_positional(self.max_len, self.llm_dim)
        self.uhd_sepparators = self.get_token_embeddings(["\n", ","])

    def get_token_embeddings(self, text):
        input_ids = self.tokenizer(text).input_ids

        with torch.no_grad():  # Optionally disable gradient calculation
            embeddings = self.llm.get_input_embeddings()(torch.tensor(input_ids).to(self.device))

        return embeddings

    def get_positional_encoding(max_seq_len, embedding_dim):
        position_encoding = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding

    def processs(self, image, text):
        #Supports just 1 image for now
        if "<image>" not in text:
            new_embeddings = self.get_token_embeddings(text)
        else:
            assert text.count("<image>") == 1
            new_embeddings = self.encode_images_no_positional_encoding(image)
            before, after = text.split("<image>")
            if len(before) > 0:
                new_embeddings = torch.cat((self.get_token_embeddings(before), new_embeddings), dim=0)
            if len(after) > 0:
                new_embeddings = torch.cat((new_embeddings, self.get_token_embeddings(after)), dim=0)

        #run the embeddings through the llm and return the result in clear text
        with torch.no_grad():
            output = self.llm(new_embeddings.unsqueeze(0))
            return self.tokenizer.decode(output[0])
        
    def forward(self, image, text):
        #Supports just 1 image for now
        if "<image>" not in text:
            new_embeddings = self.get_token_embeddings(text)
        else:
            assert text.count("<image>") == 1
            new_embeddings = self.encode_images_no_positional_encoding(image)
            before, after = text.split("<image>")
            if len(before) > 0:
                new_embeddings = torch.cat((self.get_token_embeddings(before), new_embeddings), dim=0)
            if len(after) > 0:
                new_embeddings = torch.cat((new_embeddings, self.get_token_embeddings(after)), dim=0)

        #run the embeddings through the llm and return the result in clear text
        with torch.no_grad():
            output = self.llm(new_embeddings.unsqueeze(0))
            return self.tokenizer.decode(output[0])

    def encode_images_positional_encoding(self, images, padding = True, sinusoidal_encoding = True, learned_encoding = False):
        MAX_LEN = 8

        image_tensors = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values'].to(self.device)
        #for the case where there are less than 8 images, add empty tensors
        if(padding):
            for i in range(MAX_LEN-len(images)):
                image_tensors = torch.cat((image_tensors, torch.zeros_like(image_tensors[0]).unsqueeze(0)), dim=0)
        
        with torch.no_grad(): 
            batch_features = self.vison_model(image_tensors, output_hidden_states=True)
            image_features = batch_features.hidden_states[-1]
            image_features = feature_select(image_features, "patch")
            # Positional Encoding
            if(sinusoidal_encoding):
                max_seq_len = image_features.shape[1]
                pos_encoding = self.get_positional_encoding(max_seq_len, image_features.shape[-1]).to(self.device)
                image_features += pos_encoding

        # Learned Positional Encoding
        if learned_encoding:
            image_features += self.learned_encoding_layer(image_features)

        return self.vision_projector(image_features)
    
    def images_uhd_positional_encoding(self, image):
        #lower the image with padding to 
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)
        self.encode_images_positional_encoding(splits)

    def imaged_uhd_arranged(self, image):
        resized_image = resize_with_padding(image, 336)
        splits , h , w = split_image(image)

        embeddings = self.encode_images_no_positional_encoding(splits)
        new_embeddings = []
        for i in range(h):
            for j in range(w):
                new_embeddings.append(embeddings[i*w+j])
                new_embeddings.append(self.uhd_sepparators[1])
            new_embeddings.append(self.uhd_sepparators[0])
        
        return new_embeddings
                
    
    def encode_images_no_positional_encoding(self, image_tensors):
        with torch.no_grad(): 
            batch_features = self.vison_model(image_tensors, output_hidden_states=True)
            image_features = batch_features.hidden_states[-1]
            image_features = feature_select(image_features, "patch")
        return self.vision_projector(image_features)