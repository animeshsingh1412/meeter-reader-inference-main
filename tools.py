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
    path: Path to image file.

  Returns:
    `uint8` numpy array with shape `(img_height, img_width, 3)`
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
  """
  Draw bounding box over an image with a text over it.
  Args:
    image: `uint8` numpy array with shape `(img_height, img_width, 3)`.
    rect: `flaot32` Normalized coordinates of shape `(4)`. `[X1, Y1, X2, Y2]`.
  Returns:
    image: `uint8` numpy array with shape `(img_height, img_width, 3)`. Annotated image with bounding box drawn over it.
  """
  w = image.shape[0]
  h = image.shape[1]
  x1, y1, x2, y2 = rect
  x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
  image = cv2.rectangle(image, (x1, y1), (x2,  y2), color, 1)
  cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
  return image

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):
    """
    Draw text over an image.
    Args:
      img:  `uint8` numpy array with shape `(img_height, img_width, 3)`.
      text: `string` The text that needs to be drawn over the image.
    Returns:
      img: `uint8` numpy array with shape `(img_height, img_width, 3)`. Annotated image with text drawn over it.
    """
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return img
  
