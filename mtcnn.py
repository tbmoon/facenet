import argparse
import os
from pathlib import Path

from PIL import Image
from torch_mtcnn import detect_faces


def get_dir_and_file(path):
    return os.path.join(path.parts[-2], path.parts[-1])


def valid_ext(ext):
    return ext.lower() in ['.jpg', '.jpeg', '.png']


def detect_and_store(path, final_root_dir, new_size):
    img = Image.open(path)
    bounding_boxes, landmarks = detect_faces(img)

    if len(bounding_boxes) == 0:
        print("No face detected on image", get_dir_and_file(path))

    '''
    image should only contains one valid person for the corresponding label/person
    so, we assume that bounding_boxes shape is always (1, 5)
    '''
    for a, b, c, d, _ in bounding_boxes:
        dst = os.path.join(final_root_dir, path.parent.name)
        os.makedirs(dst, exist_ok=True)
        final_name = os.path.join(dst, path.name)
        try:
            img.crop((a, b, c, d)).resize((new_size, new_size), Image.BILINEAR).save(final_name)
            print(get_dir_and_file(path), 'saved to', final_name)
        except Exception as e:
            print("Error occured when saving", get_dir_and_file(path))
            print("Error: ", str(e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detection using MTCNN')

    parser.add_argument('--root-dir', type=str, help='absolute path to dataset root dir')
    parser.add_argument('--final-dir', type=str, help='Final absolute root directory to store all the files')
    parser.add_argument('--resize', type=int, help='resize image into a given squared size', default=128)

    args = parser.parse_args()

    paths = Path(args.root_dir).glob('*/*')
    final_dir = args.final_dir

    for path in paths:
        if not valid_ext(path.suffix):
            print(get_dir_and_file(path), 'is not valid image. Expected extensions: .jpg, .jpeg, .png')
            continue

        detect_and_store(path, final_dir, args.resize)
