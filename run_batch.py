import multiprocessing
import glob
import os
import time
import json
from tqdm import tqdm
from os.path import join as pjoin, exists
import cv2

import detect_compo.ip_region_proposal as ip


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


if __name__ == '__main__':
    PSTUTS_DIR = "/Users/yiqiaoj/Workspace/data/Multimodal/CVPR2020_PsTuts"



    # initialization
    input_img_root = os.path.join(PSTUTS_DIR, 'screenshots', "0_0")
    output_root = os.path.join(PSTUTS_DIR, 'screenshots_annotated', "0_0")
    os.makedirs(output_root, exist_ok=True)

    # data = json.load(open('E:/Mulong/Datasets/rico/instances_test.json', 'r'))


    input_imgs = [os.path.join(input_img_root, image_name) for image_name in os.listdir(input_img_root)]

    key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50, 'merge-contained-ele': True,
                  'max-word-inline-gap': 10, 'max-line-ingraph-gap': 4, 'remove-top-bar': True}

    is_ip = True
    is_clf = False
    is_ocr = False
    is_merge = True

    # Load deep learning models in advance
    compo_classifier = None
    if is_ip and is_clf:
        compo_classifier = {}
        from cnn.CNN import CNN
        # compo_classifier['Image'] = CNN('Image')
        compo_classifier['Elements'] = CNN('Elements')
        # compo_classifier['Noise'] = CNN('Noise')
    ocr_model = None
    if is_ocr:
        import detect_text.text_detection as text

    # set the range of target inputs' indices
    num = 0
    start_index = 30800  # 61728
    end_index = 100000
    for input_img in input_imgs:
        image_name = os.path.basename(input_img)
        image_name = ".".join(image_name.split('.')[:-1])
        resized_height = resize_height_by_longest_edge(input_img)

        if is_ocr:
            text.text_detection(input_img, output_root, show=False)

        if is_ip:
            ip.compo_detection(input_img, output_root, key_params,  classifier=compo_classifier, resize_by_height=resized_height, show=False)

        if is_merge:
            from detect_merge import merge
            compo_path = pjoin(output_root, 'ip', f'{image_name}.json')
            ocr_path = pjoin(output_root, 'ocr', f'{image_name}.json')

            os.makedirs(os.path.dirname(compo_path), exist_ok=True)
            os.makedirs(os.path.dirname(ocr_path), exist_ok=True)

            # merge.merge(input_img, compo_path, ocr_path, output_root, is_remove_top=key_params['remove-top-bar'], show=True)
            merge.merge(input_img, compo_path, ocr_path, output_root, is_remove_bar=True, show=True)

        num += 1
