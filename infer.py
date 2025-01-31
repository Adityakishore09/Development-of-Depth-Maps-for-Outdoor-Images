"""
This code is for inferring the results on basic DepthAnything and the trained DepthAnything
"""

from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from PIL import Image, ImageChops
import torch
import matplotlib.pyplot as plt

"""Arguments"""

# Image path, name and if want to save the depth map
path = '/data2/dse411a/project3/team2/sample_images/day_busy.jpg'
image_name = 'train_day_busy'
image = Image.open(path)
save_res = False

# Loading the models from Hugging Face
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
base_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
own_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# Load encoder weights
encoder_root = '/data2/dse411a/project3/team2/Depth_Anything_v2/try_encoder/First_code/run2/enc_Encoder_fromscratch_15_epochs.pth'
own_model.backbone.load_state_dict(torch.load(encoder_root, weights_only= False))



""" Function to generate the depth maps. Uses the process outlined by the HF model card for DepthAnything"""
def get_depth_map(image, model, model_type, image_name, save_res):
    inputs = image_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)])
    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    if save_res:
        save_name = f'{model_type}_{image_name}.png'
        print("Saving image ...")
        depth.save(save_name)
    
    return depth

# Base model
base_depth = get_depth_map(image, base_model, 'base', image_name, save_res)
# Own model
own_depth = get_depth_map(image, own_model, 'own', image_name, save_res)

"""Function to display and compare the depth maps"""
def display_depth_maps(img, base, own, image_name):
    
    diff = ImageChops.subtract(own, base)

    plt.figure(figsize= (20, 16))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.set_axis_off()
    
    ax2.imshow(base)
    ax2.set_title('Base Model Depth Image')
    ax2.set_axis_off()

    ax3.imshow(own)
    ax3.set_title('Our Model Depth Image')
    ax3.set_axis_off()

    ax4.imshow(diff)
    ax4.set_title('Depth Difference')
    ax4.set_axis_off()

    plt.axis('off')
    plt.tight_layout(pad= 0.3, w_pad= 0.2, h_pad= 0.2)
    plt.show() # Does not work on server :(
    plt.savefig(f'{image_name}_comparison.png', bbox_inches = 'tight')

display_depth_maps(image, base_depth, own_depth, image_name)