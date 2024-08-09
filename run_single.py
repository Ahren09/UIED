import traceback
from os.path import join as pjoin
import cv2
import os
import os.path as osp
import numpy as np


def resize_image(dimensions):
    """
    Resize the image dimensions so that the shortest side is at most 768 pixels,
    while maintaining the aspect ratio.

    Parameters:
    dimensions (tuple): A tuple (height, width) representing the current dimensions of the image.

    Returns:
    tuple: A tuple (new_height, new_width) representing the resized dimensions of the image.
    """
    height, width = dimensions

    # Determine the scale factor based on the shortest side
    if height < width:
        if height > 768:
            scale_factor = 768 / height
            new_height = 768
            new_width = int(width * scale_factor)
        else:
            new_height = height
            new_width = width
    else:
        if width > 768:
            scale_factor = 768 / width
            new_width = 768
            new_height = int(height * scale_factor)
        else:
            new_height = height
            new_width = width

    return (new_height, new_width)


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))

def resize_width_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if width > height:  # width / height > 1.
        return resize_length
    else:  # width / height <= 1.
        return int(resize_length * (width / height))


def color_tips():
    color_map = {'Text': (0, 0, 255), 'Compo': (0, 255, 0), 'Block': (0, 255, 255), 'Text Content': (255, 0, 255)}
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, 'Text', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, 'Non-text Compo', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Compo's Text Content", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('colors', board)


def annotate_image(image_path: str):

    '''
        ele:min-grad: gradient threshold to produce binary map         
        ele:ffl-block: fill-flood threshold
        ele:min-ele-area: minimum area for selected elements 
        ele:merge-contained-ele: if True, merge elements contained in others
        text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
        text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

        Tips:
        1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
        2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
        3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
        4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

        mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
        web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    '''
    key_params = {'min-grad':15, 'ffl-block':5, 'min-ele-area':200,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}




    # set input image path
    input_path_img = image_path
    output_root = 'data/output'


    # resized_height = resize_height_by_longest_edge(input_path_img, resize_length=800)
    # resized_width = resize_width_by_longest_edge(input_path_img, resize_length=768)

    org = cv2.imread(input_path_img)
    height, width = org.shape[:2]
    resized_height, resized_width = resize_image((height, width))


    color_tips()

    is_ip = True
    is_clf = False
    is_ocr = True
    is_merge = False

    if is_ocr:
        import detect_text.text_detection as text
        os.makedirs(pjoin(output_root, 'ocr'), exist_ok=True)
        text.text_detection(input_path_img, output_root, show=True, method='google')

    if is_ip:
        import detect_compo.ip_region_proposal as ip
        os.makedirs(pjoin(output_root, 'ip'), exist_ok=True)
        # switch of the classification func
        classifier = None
        if is_clf:
            classifier = {}
            from cnn.CNN import CNN
            # classifier['Image'] = CNN('Image')
            classifier['Elements'] = CNN('Elements')
            # classifier['Noise'] = CNN('Noise')
        ip.compo_detection(input_path_img, output_root, key_params,
                           classifier=classifier, resize_by_height=resized_height, resize_by_width=resized_width, show=False)

    if is_merge:
        import detect_merge.merge as merge
        os.makedirs(pjoin(output_root, 'merge'), exist_ok=True)
        name = input_path_img.split('/')[-1][:-4]
        compo_path = pjoin(output_root, 'ip', str(name) + '.json')
        ocr_path = pjoin(output_root, 'ocr', str(name) + '.json')
        merge.merge(input_path_img, compo_path, ocr_path, pjoin(output_root, 'merge'),
                    is_remove_bar=key_params['remove-bar'], is_paragraph=key_params['merge-line-to-paragraph'], show=True)


if __name__ == "__main__":
    PSTUTS_DIR = osp.expanduser("~/Workspace/data/Multimodal/CVPR2020_PsTuts")
    all_video_names = sorted(os.listdir(osp.join(PSTUTS_DIR, "video_clips")))
    for video_id in range(1000):
        for segment_id in range(1000):
            video_name = f"{video_id}_{segment_id}.mp4"
            video_path = osp.join(PSTUTS_DIR, "video_clips", video_name)
            if osp.exists(video_path):

                # TODO: detect key frames for the timestamp
                screenshot_path = osp.join(PSTUTS_DIR, "screenshots", f"{video_id}_{segment_id}")
                screenshot_names = os.listdir(screenshot_path)
                timestamps = sorted([float(name.split('_')[-1].split('.png')[0]) for name in screenshot_names])

                for timestamp in timestamps:
                    try:
                        screenshot_path = osp.join(PSTUTS_DIR, "screenshots", f"{video_id}_{segment_id}", f"screenshot_{timestamp}.png")
                        annotate_image(screenshot_path)

                    except Exception as e:
                        print(f"Fail to same screenshot at t = {timestamp} secs")
                        traceback.print_exc()
                        break



