from rembg import remove
from PIL import Image

input_path = "./img.jpg"  # Corrected file name
output_path = "./out.png"

# Open the input image
with Image.open(input_path) as img:
    # Convert the image to RGBA (to ensure it has an alpha channel)
    img = img.convert("RGBA")
    
    # Remove the background
    output = remove(img)
    
    # Save the output image
    output.save(output_path, format="PNG")