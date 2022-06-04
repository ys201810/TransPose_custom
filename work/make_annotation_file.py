# coding=utf-8
"""
keypoint用のアノテーションファイルを作成する。
jsonファイルで、keys()は['info', 'licenses', 'images', 'annotations', 'categories']
info/licenses/categoriesは固定。imagesは画像の数の辞書のリスト。annotationsはアノテーションの数の辞書のリスト。
imagesのkeys()は、['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
annotationsのkeys()は、['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
"""
import os
import cv2
import json
import pickle
import random
import numpy as np


def make_info():
    return {'description': '2022/05/26',
            'url': 'https://nineedge.co.jp/',
            'version': '1.0',
            'year': 2022,
            'contributor': 'Nine Edge'
            }


def make_licenses():
    return [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/',
             'id': 1,
             'name': 'Attribution-NonCommercial-ShareAlike License'}]


def make_categories():
    return [{'supercategory': 'person',
             'id': 1,
             'name': 'person',
             'keypoints': ['nose',
                           'left_eye',
                           'right_eye',
                           'left_ear',
                           'right_ear',
                           'left_shoulder',
                           'right_shoulder',
                           'left_elbow',
                           'right_elbow',
                           'left_wrist',
                           'right_wrist',
                           'left_hip',
                           'right_hip',
                           'left_knee',
                           'right_knee',
                           'left_ankle',
                           'right_ankle',
                           'right_toe',  # 追加
                           'left_toe',   # 追加
                           'top_bad'],   # 追加
             'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
                          [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
                          [2, 4], [3, 5], [4, 6], [5, 7],
                          [17, 16], [18, 15]]  # つま先とバッドの先を追加
             }]


def make_images(i, file_name, width, height):
    image_dict = {}
    image_dict['license'] = 1
    image_dict['file_name'] = file_name
    image_dict['coco_url'] = ''
    image_dict['width'] = width
    image_dict['height'] = height
    image_dict['date_captured'] = '2022-05-26 00:00:00'
    image_dict['flickr_url'] = ''
    image_dict['id'] = i  # image seq
    return image_dict


def make_annotations(i, results, bbox, add_keypoints):
    if len(results + add_keypoints) != 60:
        exit('keypoints list num is wrong. correct:{} current:{}'.format(60, len(results + add_keypoints)))
    annotation_dict = {}
    annotation_dict['segmentation'] = [[]]
    annotation_dict['num_keypoints'] = 20  # keypointsのx,y,vのv!=0の数を入力。transposeそのままなら17。 左右のつま先とバッドの先を追加するので20
    annotation_dict['area'] = 1  # segmentationのピクセル数(1以上がセットされていないと学習できない。。)
    annotation_dict['iscrowd'] = 0  # 群衆の時にセット
    annotation_dict['keypoints'] = results + add_keypoints  # 17 * 3のリスト categoriesのkeypointsの順にx, y, v(0: not labeled, v=1: labeled but not visible, and v=2 labeled and visible)
    annotation_dict['image_id'] = i
    annotation_dict['bbox'] = [int(val) for val in bbox]  # person bbox
    annotation_dict['category_id'] = 1
    annotation_dict['id'] = i  # annotation_seq
    return annotation_dict


def main():
    # 学習用jsonファイル作成 'images', 'annotations', '']
    train_final_json = {}
    train_final_json['info'] = make_info()
    train_final_json['licenses'] = make_licenses()
    train_final_json['categories'] = make_categories()

    # 検証用jsonファイル作成 'images', 'annotations', '']
    valid_final_json = {}
    valid_final_json['info'] = make_info()
    valid_final_json['licenses'] = make_licenses()
    valid_final_json['categories'] = make_categories()

    # TransPoseの予測結果のロード
    annotation_file = os.path.join('data', 'keypoint_results.pkl')
    with open(annotation_file, 'rb') as inf:
        annotations = pickle.load(inf)

    # coco annotation toolの結果のロード
    annotation_result_file = os.path.join('data', 'coco_annotation_tool_results.json')
    with open(annotation_result_file, 'r') as inf:
        annotation_results = json.load(inf)

    # key: 画像名, valueに右のつま先, 左のつま先, バッドの先端のリストの辞書を作成
    id_file_dict = {}
    for val in annotation_results['images']:
        id_file_dict[val['id']] = os.path.basename(val['path'])
    add_keypoint_dict = {}
    for val in annotation_results['annotations']:
        file_name = id_file_dict[val['image_id']]
        add_keypoint_dict[file_name] = val['keypoints']

    train_image_infos = []
    valid_image_infos = []
    train_annotation_infos = []
    valid_annotation_infos = []
    image_dir = os.path.join('data', 'images')
    # 1つの結果に1つのアノテーション
    for i, vals in enumerate(sorted(annotations.items())):
        # images作成
        image_file, results = vals
        image_file_name = os.path.basename(image_file)
        # 追加アノテーションが存在しなければスキップ
        if image_file_name not in add_keypoint_dict.keys():
            continue
        else:
            # 追加アノテーションがある場合、右のつま先、左のつま先、バッドの先のkeypointデータを取得
            add_keypoints = add_keypoint_dict[image_file_name]

        height, width = cv2.imread(os.path.join(image_dir, image_file_name)).shape[:2]
        # image_infos.append(make_images(i, image_file_name, width, height))

        # annotations作成
        bbox = results['bbox']
        keypoints = results['keypoints']
        keypoints = np.insert(keypoints, 2, 2, axis=2)  # x, y, vのvに2: labeled and visibleを固定でセット
        keypoints = np.ravel(keypoints.astype(int)).tolist()  # int型にして51の要素のリストに変換
        # annotation_infos.append(make_annotations(i, keypoints, bbox))

        val_rate = 0.2
        # valに利用する割合でtrain/valを振り分け
        random_split = random.randint(0, 9)
        if random_split > val_rate * 10:
            train_image_infos.append(make_images(i, image_file_name, width, height))
            train_annotation_infos.append(make_annotations(i, keypoints, bbox, add_keypoints))
        else:
            valid_image_infos.append(make_images(i, image_file_name, width, height))
            valid_annotation_infos.append(make_annotations(i, keypoints, bbox, add_keypoints))

    train_final_json['images'] = train_image_infos
    train_final_json['annotations'] = train_annotation_infos

    valid_final_json['images'] = valid_image_infos
    valid_final_json['annotations'] = valid_annotation_infos

    with open('data/annotations/baseball_keypoints_train.json', 'w') as outf:
        json.dump(train_final_json, outf)

    with open('data/annotations/baseball_keypoints_val.json', 'w') as outf:
        json.dump(valid_final_json, outf)


if __name__ == '__main__':
    print(212123)
    main()
