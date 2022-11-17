import warnings
from model import create_model
import numpy as np
import os.path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import joblib

from align import AlignDlib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.manifold import TSNE


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

    def image_name(self):
        return self.file


def load_metadata(path):
    metadata = []
    for i in sorted(os.listdir(path)):
        for f in sorted(os.listdir(os.path.join(path, i))):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.bmp':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)


def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[..., ::-1]


def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


def show_pair(idx1, idx2):
    plt.figure(figsize=(8, 3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()))


nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('D://GITHUB//CNN-Face-Detection-SVM//Buoc2//weights//nn4.small2.v1.h5')


metadata = load_metadata('D:\GITHUB\CNN-Face-Detection-SVM\Buoc1\image')

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('D:\GITHUB\CNN-Face-Detection-SVM\Buoc2\models\shape_predictor_68_face_landmarks.dat')

# Load an image of a person
jc_orig = load_image(metadata[0].image_path())  # 140
# jc_orig = load_image(metadata[90].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(
    96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.subplot(131)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)
plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()),
                                      bb.width(), bb.height(), fill=False, color='red'))


# Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned)

plt.show()

embedded = np.zeros((metadata.shape[0], 128))

file_log = open("err_log.txt", "w")


for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    # print(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    if img is not None:
        pass
    else:
        print(m.image_path())
        file_log.writelines(m.image_path()+"\n")

file_log.close()
