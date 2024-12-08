from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the pre-trained model, feature extractor, and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Prediction function
def predict_step(image_paths, batch_size=4):
    """
    Predict captions for a batch of images.

    Args:
        image_paths (list of str): List of file paths to the images.
        batch_size (int): Number of images to process in a single batch.

    Returns:
        dict: Dictionary with image file names as keys and captions as values.
    """
    captions = {}
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = []
        for image_path in batch_paths:
            if not os.path.exists(image_path):
                print(f"Warning: File not found - {image_path}")
                continue
            try:
                i_image = Image.open(image_path)
                if i_image.mode != "RGB":
                    i_image = i_image.convert(mode="RGB")
                images.append(i_image)
            except Exception as e:
                print(f"Error processing file {image_path}: {e}")
                continue

        if not images:
            continue

        # Preprocess images
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Generate captions
        with torch.no_grad():
            output_ids = model.generate(pixel_values, **gen_kwargs)

        # Decode captions
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        # Map captions to their respective file names
        for path, caption in zip(batch_paths, preds):
            captions[path] = caption

    return captions

# Function to display images with captions
def display_images_with_captions(image_captions):
    """
    Display images alongside their generated captions.

    Args:
        image_captions (dict): Dictionary with image file names as keys and captions as values.
    """
    for image_path, caption in image_captions.items():
        try:
            img = Image.open(image_path)
            plt.figure()
            plt.imshow(img)
            plt.axis("off")
            plt.title(caption)
            plt.show()
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")

# Save captions to a text file
def save_captions_to_file(image_captions, output_file="captions.txt"):
    """
    Save the generated captions to a text file.

    Args:
        image_captions (dict): Dictionary with image file names as keys and captions as values.
        output_file (str): File name to save the captions.
    """
    with open(output_file, "w") as f:
        for image_path, caption in image_captions.items():
            f.write(f"{image_path}: {caption}\n")
    print(f"Captions saved to {output_file}")

# Example usage
if __name__ == "__main__":
    image_paths = [
        "/content/monkey-6952630_1280.jpg",  # Replace with valid image paths
        "/content/seedless_fruits.jpg",       # Replace with valid image paths
        "/content/._33108590_d685bfe51c.jpg",       # Example of a missing file
    ]

    captions = predict_step(image_paths)
    display_images_with_captions(captions)
    save_captions_to_file(captions)
