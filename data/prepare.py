import os
import cv2
from tqdm import tqdm

def make():
    image_list = os.listdir(os.path.join('0'))
    for one in tqdm(image_list):
        read_path = os.path.join('0',one)
        image_content = cv2.imread(read_path)
        if(image_content.shape[0] >= 400 and image_content.shape[1] >= 400):
            # clear_write_path = os.path.join('SRGAN', 'ClearImage', one)
            fuzzy_write_path = os.path.join('SRGAN', 'MoreFuzzyImage', one)
            # clear_image_content = cv2.resize(image_content,(384,384),interpolation = cv2.INTER_CUBIC)
            fuzzy_image_content = cv2.resize(image_content,(48,48))
            fuzzy_image_content = cv2.resize(image_content,(96,96))
            # cv2.imwrite(clear_write_path,clear_image_content)
            cv2.imwrite(fuzzy_write_path,fuzzy_image_content)


if __name__ == '__main__':
    make()
