# coding=utf-8
import cv2
import json
from work.cv_utils import draw_circle

def main():
    # アノテーションのロード
    annotation_file = 'data/annotations/person_keypoints_train2017.json'
    with open(annotation_file, 'r') as inf:
        annotation = json.load(inf)

    target_index = 60862
    image_info = annotation['images'][target_index]
    annotation_infos = [val for val in annotation['annotations'] if val['image_id'] == image_info['id']]

    print(image_info)
    print(annotation_infos)

    # 画像のロード
    img = cv2.imread('data/images/train2017/' + image_info['file_name'])
    # img = cv2.imread('data/images/test.jpg')

    # キーポイントの取得
    POINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
              "right_knee", "left_ankle", "right_ankle"]

    for annotation_info in annotation_infos:
        keypoints = annotation_info['keypoints']
        print(keypoints)

        annotated_img = img.copy()

        for i in range(17):
            point_height = keypoints[1 + 3 * i]
            point_width = keypoints[3 * i]
            # center = (point_height, point_width)  # centerは(x, y)で渡す
            center = (point_width, point_height)
            annotated_img = draw_circle(annotated_img, center, 3, color=(255, 255, 0), thickness=1)  # BGR

    cv2.imwrite('a.jpg', annotated_img)


if __name__ == '__main__':
    main()
