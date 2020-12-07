from skimage import feature
import numpy as np
import cv2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    @staticmethod
    def create_circular_mask(h, w, center=None, radius=None):

        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        hist = [float(elem) for elem in hist]
        return hist, lbp

    def describe_regions(self, image, window_size=None, eps=1e-7):

        image = cv2.GaussianBlur(image, (5, 5), 0)

        mean, std = cv2.meanStdDev(image)

        retval, thr = cv2.threshold(image, int(mean), 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thr = cv2.dilate(thr, kernel, iterations=3)

        image = image*thr

        norm_img = np.zeros((120, 120))
        image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)


        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="default")

        if window_size is not None:
            lbp_chunks = self.chunkify(lbp, block_width=window_size[0], block_height=window_size[1])
        else:
            lbp_chunks = self.chunkify(lbp)
        hist = []
        for chunk in lbp_chunks:
            (hist_chunk, _) = np.histogram(chunk.ravel(),
                                     bins=np.arange(0, self.numPoints + 3),
                                     range=(0, self.numPoints + 2))
            # normalize the histogram
            hist_chunk = hist_chunk.astype("float")
            hist_chunk /= (hist_chunk.sum() + eps)
            # return the histogram of Local Binary Patterns
            hist_chunk = [float(elem) for elem in hist_chunk]
            hist += hist_chunk
        return hist

    @staticmethod
    def chunkify(img, block_width=5, block_height=5):
        x_len = int(img.shape[0] / block_width)
        y_len = int(img.shape[1] / block_height)
        chunks = []

        for i in range(x_len):
            for j in range(y_len):
                chunks.append(img[j*block_height:(j+1)*block_height, i*block_width:(i+1)*block_width])

        return chunks