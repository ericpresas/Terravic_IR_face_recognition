import cv2
from LBP import LocalBinaryPatterns
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle


def get_features(paths, dataset_faces, desc):
    data = []
    num_paths = len(paths)
    for idx_path in tqdm(range(num_paths)):
        path_image = paths[idx_path]
        img = cv2.imread(path_image, 0)
        if img is not None:
            faces_det = dataset_faces.detectMultiScale(img.astype('uint8'))
            if len(faces_det) > 0:
                img_face = img
                valid = True
                for (x, y, w, h) in faces_det:
                    if w >= 120 and h >= 120:
                        img_face = img[y:y + h, x:x + w]

                        # dsize
                        dsize = (120, 120)
                        # resize image
                        img_face = cv2.resize(img_face, dsize)
                    else:
                        valid = False
                if valid:
                    hist = desc.describe_regions(img_face, window_size=[10, 10])
                    #hist, lbp_img = desc.describe(img_face)
                    #plt.imshow(lbp_img, cmap='gray')
                    #plt.show()
                    data.append(hist)
    return data

path = 'terravic_facial_infrared_dataset/'
dataset_faces = cv2.CascadeClassifier('haar_cascade/haarcascade_frontalface_default.xml')  # Face dataset

classes = {}
faces_paths = [x[0] for x in os.walk(path)]
faces_paths = faces_paths[1:]
for i, face_path in enumerate(faces_paths):
    for fn in next(os.walk(face_path))[2]:
        class_ = str(i+1)
        if class_ not in classes:
            classes[class_] = {
                "paths": [f'{face_path}/{fn}']
            }
        else:
            classes[class_]['paths'].append(f'{face_path}/{fn}')

    train, test = train_test_split(np.array(classes[class_]['paths']), test_size=0.5)
    validation, test = train_test_split(test, test_size=0.2)
    classes[class_]['train'] = list(train)
    classes[class_]['validation'] = list(validation)
    classes[class_]['test'] = list(test)

    if len(classes[class_]['train']) < 500:
        del classes[class_]


desc = LocalBinaryPatterns(16, 2)
labels_train = []
data_train = []
labels_test = []
data_test = []
labels_val = []
data_val = []
count = 0

for label, item in classes.items():
    if count < 10:
        print(f'Class: {count+1}')
        train_class = get_features(item['train'], dataset_faces, desc)
        labels_train_class = [int(count+1) for i in range(len(train_class))]
        test_class = get_features(item['test'], dataset_faces, desc)
        labels_test_class = [int(count+1) for i in range(len(test_class))]
        val_class = get_features(item['validation'], dataset_faces, desc)
        labels_val_class = [int(count + 1) for i in range(len(val_class))]

        data_train += train_class
        labels_train += labels_train_class
        data_test += test_class
        labels_test += labels_test_class
        data_val += val_class
        labels_val += labels_val_class

    count += 1

with open('terravic_lbph_features/train.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(data_train, filehandle)

with open('terravic_lbph_features/labels_train.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(labels_train, filehandle)

with open('terravic_lbph_features/test.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(data_test, filehandle)

with open('terravic_lbph_features/labels_test.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(labels_test, filehandle)

with open('terravic_lbph_features/validation.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(data_val, filehandle)

with open('terravic_lbph_features/labels_validation.data', 'wb') as filehandle:
    # store the data as binary data stream
    pickle.dump(labels_val, filehandle)