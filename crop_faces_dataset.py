import cv2
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os


class IRFacesDataset(object):
    def __init__(self, path, test_size=0.2, validation_size=0.5):
        self.face_detector = cv2.CascadeClassifier(
            'haar_cascade/haarcascade_frontalface_default.xml')  # Face dataset

        self.paths = {}
        faces_paths = [x[0] for x in os.walk(path)]
        faces_paths = faces_paths[1:]
        for i, face_path in enumerate(faces_paths):
            for fn in next(os.walk(face_path))[2]:
                class_ = str(i + 1)
                if class_ not in self.paths:
                    self.paths[class_] = {
                        "paths": [f'{face_path}/{fn}']
                    }
                else:
                    self.paths[class_]['paths'].append(f'{face_path}/{fn}')

            train, test = train_test_split(np.array(self.paths[class_]['paths']), test_size=validation_size)
            validation, test = train_test_split(test, test_size=test_size)
            for a in train:
                b = list(filter(lambda x: x==a, test))
                if len(b)>0:
                    print(f'{a} duplicated')
            self.paths[class_]['train'] = list(train)
            self.paths[class_]['test'] = list(test)
            self.paths[class_]['validation'] = list(validation)

    def crop_dataset(self, path_output):
        for label, item in self.paths.items():
            print(f'Class: {label}')
            class_out_path = f'{path_output}{label}'
            try:
                os.mkdir(f'{class_out_path}')
                print(f'Created path class {label}')
            except FileExistsError as err:
                print(f'{class_out_path} Already exists')

            try:
                os.mkdir(f'{class_out_path}/train')
                print(f'Created path class {label} train')
            except FileExistsError as err:
                print(f'{class_out_path}/train Already exists')
            try:
                os.mkdir(f'{class_out_path}/test')
                print(f'Created path class {label} test')
            except FileExistsError as err:
                print(f'{class_out_path}/test Already exists')

            try:
                os.mkdir(f'{class_out_path}/validation')
                print(f'Created path class {label} validation')
            except FileExistsError as err:
                print(f'{class_out_path}/validation Already exists')

            self.crop_class(item['train'], f'{path_output}{label}/train/')
            self.crop_class(item['test'], f'{path_output}{label}/test/')
            self.crop_class(item['validation'], f'{path_output}{label}/validation/')

    def crop_class(self, paths, path_output):
        num_paths = len(paths)
        for idx_path in tqdm(range(num_paths)):
            path_image = paths[idx_path]
            img = cv2.imread(path_image, 0)
            if img is not None:
                faces_det = self.face_detector.detectMultiScale(img.astype('uint8'))
                if len(faces_det) > 0:
                    img_face = img
                    valid = True
                    for (x, y, w, h) in faces_det:
                        if w >= 120 and h >= 120:
                            img_face = img[y:y + h, x:x + w]
                            # dsize
                            dsize = (224, 224)
                            # resize image
                            img_face = cv2.resize(img_face, dsize)
                        else:
                            valid = False
                    if valid:
                        cv2.imwrite(f'{path_output}{idx_path}.jpg', img_face)


if __name__ == '__main__':
    dataset = IRFacesDataset(path='terravic_facial_infrared_dataset', test_size=0.2, validation_size=0.5)
    dataset.crop_dataset(path_output='crop_terravic/')