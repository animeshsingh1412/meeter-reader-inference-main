import tensorflow as tf
import cv2
import numpy as np

from PIL import Image
from six import BytesIO
import tensorflow as tf

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: a file path.

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """

  # image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  # (im_height, im_width, _) = image.shape
  # return np.array(image).reshape(
  #     (im_height, im_width, 3)).astype(np.uint8)

  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def draw_bb(image, rect, text, color):
  x1, y1, x2, y2 = rect
  image = cv2.rectangle(image, (x1, y1), (x2,  y2), color, 1)
  cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
  return image
  
