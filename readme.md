### TransPose_custom
TransPoseのコードを自前データでfine tuningできるようにする。  

### package(version)
python>=3.7
torch>=1.6.0
torchvision>=0.7.0
others refer to requirements.txt

### 使い方
1. 学習用の画像を用意する。
work/data/images/配下に学習に利用したい動画を用意する。  
動画から用意する場合は、work/data/movies/配下に動画を格納し、以下を実行する。
対象とする動画は以下のスクリプトのtarget_movie_nameを変更する。

```
python make_images_from_movie.py
```
 
2. TransPoseでkeypointの座標とバウンディングボックスを予測する。
work/data/images/配下の画像に対し、TransPoseで予測を実行する。  
これを実行すると、work/data/results/配下にkeypointがプロットされた画像が出力される。  
また、work/data/配下にkeypoint_results.pklというpickleファイルが出力される。  
これは画像パスがkey、keypointとbboxがdictの辞書ファイルとなっており、これを用いてアノテーションデータを作成する。  

```
python pred_images.py
```

作成されるkeypoint_results.pklの例

```
a['data/images/15674226503903_55.jpg']
{'keypoints': array([[[492.17365, 170.1361 ],
        [501.86304, 165.32994],
        [485.36862, 164.02638],
        [515.8544 , 169.39442],
        [472.5214 , 166.6905 ],
        [525.507  , 202.91425],
        [468.20508, 204.02875],
        [543.9832 , 246.96095],
        [499.45682, 229.58784],
        [567.7238 , 219.1289 ],
        [559.7949 , 211.84406],
        [516.59564, 310.1331 ],
        [469.69473, 311.11163],
        [543.2078 , 384.65918],
        [451.04272, 387.8206 ],
        [562.60785, 449.9644 ],
        [424.01755, 465.62823]]], dtype=float32), 
'bbox': [395.7991, 121.810684, 214.31094, 391.10748]}
```

keypointsは、17個のkeypointの(x, y)座標、bboxは人物の左上の座標(x1, y1)と右下の座標(x2, y2)が(x1, y1, x2, y2)という形でセットされる。  

3. 学習用のアノテーションファイルを作成する。
keypoint_results.pklを用いて、アノテーション用のファイルを作成する。  
アノテーションファイルはtrainとvalを作成する必要があり、以下のコマンドで作成する。  
trainとvalに利用する画像の割合は[ここ](https://github.com/ys201810/TransPose_custom/blob/39e4a6ef6a7c09398a254f971a2cde18d991d12b/work/make_annotation_file.py#L126)で調整

```
python make_annotation_file.py
```

4. 画像ファイル名を変更する。
pycocotoolの中で、学習・検証用の画像ファイル名が連番に置き換わる。  
画像ファイルをこの連番で探しにいくため、画像ファイル自体の名称を変える必要がある。  
(本当はそのままのファイル名指定でもできるような気がするが、cocoの機能にこの連番のindex番号でアノテーションや画像を取得する機能が実装されており、修正すると動かなくなる可能性があるため、画像ファイルを修正する方針でいく。)  

画像ファイルと連番の対応づけは、tools/train.pyの中のtrain_datasetやvalid_datasetの作成時に作成される。
lib/dataset/coco.pyの[self.db](https://github.com/ys201810/TransPose_custom/blob/39e4a6ef6a7c09398a254f971a2cde18d991d12b/lib/dataset/coco.py#L106)に対応づけが作られる。  
この中身をdumpし、data/images/配下のファイル名を修正する。

5. 学習を実行する。
学習はGPUがないとnvccコマンドでエラーになる。  
そのため、CPU環境では学習できない。  

```
python tools/train.py --cfg experiments/coco/transpose_r/TP_R_256x192_d256_h1024_enc4_mh8.yaml
```

#### 学習データ収集方法
[これ](https://dev.classmethod.jp/articles/making-datasets-for-pose-estimation-by-using-coco-annotator/https://dev.classmethod.jp/articles/making-datasets-for-pose-estimation-by-using-coco-annotator/)を参考にアノテーションを集める。
