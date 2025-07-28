import os
import sys
from skimage.io import imread, imsave
from skimage import img_as_ubyte
from sigver.preprocessing.normalize import preprocess_signature

if len(sys.argv) < 3:
    print("Usage: save_preprocessed.py <output_dir> <image1> [image2 ...]")
    sys.exit(1)

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
for img_path in sys.argv[2:]:
    img = img_as_ubyte(imread(img_path, as_gray=True))
    processed = preprocess_signature(img, (840, 1360))
    name = os.path.splitext(os.path.basename(img_path))[0] + "_py.png"
    imsave(os.path.join(out_dir, name), processed)
    print("Saved", name)
