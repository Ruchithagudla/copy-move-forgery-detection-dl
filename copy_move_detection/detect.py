from pathlib import Path
from . import image_object  # ✅ relative import

def detect(input_path, output_path, block_size=32):
    """
    Detects an image under a specific directory
    """
    input_path = Path(input_path)
    filename = input_path.name
    output_path = Path(output_path)

    if not input_path.exists():
        print("Error: Source image did not exist.")
        exit(1)
    elif not output_path.exists():
        print("Error: Output directory did not exist.")
        exit(1)

    # ✅ Call class from local module
    single_image = image_object.ImageObject(input_path, filename, output_path, block_size)
    image_result_path = single_image.run()

    print("Done.")
    return image_result_path
