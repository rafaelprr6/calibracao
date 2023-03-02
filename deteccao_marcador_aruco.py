import cv2
from cv2 import aruco
import numpy as np

##### Descricao #####
# Este programa detecta o id e as quinas de um aruco.
# Ao final uma imagem é mostrada destacando o aruco,
# como tambem seu id.

# Funcao de redimensionamento da imagem


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# Dicionario Aruco
marker_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

param_markers = aruco.DetectorParameters_create()

img = cv2.imread("aruco.jpeg")
img = resize(img, width=600)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Esta funcao detecta as quinas e o id em um aruco
marker_corners, marker_IDs, reject = aruco.detectMarkers(
    gray_img, marker_dict, parameters=param_markers
)

if marker_corners:
    for ids, corners in zip(marker_IDs, marker_corners):
        cv2.polylines(
            img, [corners.astype(np.int32)], True, (0,
                                                    255, 255), 4, cv2.LINE_AA
        )
        corners = corners.reshape(4, 2)
        corners = corners.astype(int)
        top_right = corners[0].ravel()
        top_left = corners[1].ravel()
        bottom_right = corners[2].ravel()
        bottom_left = corners[3].ravel()
        cv2.putText(
            img,
            f"id: {ids[0]}",
            top_right,
            cv2.FONT_HERSHEY_PLAIN,
            1.3,
            (200, 100, 0),
            2,
            cv2.LINE_AA,
        )
        # print(ids, "  ", corners)

cv2.imshow("frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
