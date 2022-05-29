# coding=utf-8
import os
import cv2


def main():
    movie_dir = os.path.join('data', 'movies')
    target_movie_name = '15674226503903.MP4'
    output_dir = os.path.join('data', 'images')
    cap = cv2.VideoCapture(os.path.join(movie_dir, target_movie_name))

    i = 0
    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            break
        output_file = os.path.join(output_dir, target_movie_name.split('.')[0] + '_' + str(i) + '.jpg')
        cv2.imwrite(output_file, image)
        i += 1


if __name__ == '__main__':
    main()
