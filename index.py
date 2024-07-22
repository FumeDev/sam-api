from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class SamModelHandler:
    def __init__(self, model_type, checkpoint_path, device, points_per_side=32):
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.model, points_per_side=points_per_side, box_nms_thresh=0.1)
    
    def generate_masks(self, image_rgb):  
        return self.mask_generator.generate(image_rgb) # Assuming SamModelHandler is in a separate file

app = Flask(__name__)

# Initialize the SamModelHandler
# You might want to adjust these parameters based on your needs
model_handler = SamModelHandler(
    model_type="vit_h",
    checkpoint_path="sam_vit_h_4b8939.pth",
    device="cuda" if torch.cuda.is_available() else "cpu",
    points_per_side=32
)

@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    # Read the image file
    image = Image.open(io.BytesIO(file.read()))
    
    # Convert image to RGB (SAM expects RGB input)
    image_rgb = image.convert("RGB")
    
    # Convert to numpy array
    image_array = np.array(image_rgb)
    
    # Generate masks
    masks = model_handler.generate_masks(image_array)
    
    # Convert masks to a serializable format
    serializable_masks = [
        {
            "segmentation": mask["segmentation"].tolist(),
            "area": int(mask["area"]),
            "bbox": mask["bbox"],
            "predicted_iou": float(mask["predicted_iou"]),
            "point_coords": mask["point_coords"] if mask["point_coords"] is not None else None,
            "stability_score": float(mask["stability_score"]),
            "crop_box": mask["crop_box"],
        }
        for mask in masks
    ]
    
    return jsonify({"masks": serializable_masks})

if __name__ == '__main__':
    app.run(debug=True, port=5557)