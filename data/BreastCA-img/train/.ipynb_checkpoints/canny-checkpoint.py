import cv2
import os

def create_sketch(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to smooth the image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny to detect edges
    edges = cv2.Canny(blurred_image, 
                      threshold1=40,
                      threshold2=120,
                      apertureSize=3, # can only take 3-7
                      L2gradient=True)
    
    # Invert the edges to create a sketch effect
    inverted_edges = cv2.bitwise_not(edges)
    
    # Optionally, you can blend with the original for more artistic style
    # sketch = cv2.divide(gray_image, 255 - edges, scale=256)

    return inverted_edges

def create_edge_maps(input_dir, sketch_dir):
    inputs_files = os.listdir(input_dir)
    for filename in inputs_files:
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_sketch = create_sketch(img)
        sketch_img_path = os.path.join(sketch_dir, filename)
        cv2.imwrite(sketch_img_path, img_sketch)
        
input_dir = '/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/all/us/'
sketch_dir = '/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/all/canny/'
create_edge_maps(input_dir, sketch_dir)
