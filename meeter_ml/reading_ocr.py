import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import os.path
import sys

__pdoc__ = {
    'main': False
}

def load_tflite_model(model_path):
  """
  Load the tf-lite model into memory.
  Args:
    model_path: Path to tensorflow lite model for `ReadingOCR`
  Returns:
    model: Interpreter for tensorflow lite `ReadingOCR` model
  """
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter

def prepare_input(image_path, input_size, bbox=None):
  """
  Preprocessing for the input image for OCR network. It converts the image into greyscale and crops it by given `bbox`.
  If `bbox` is None, then no cropping is performed. The image is resized to `input_size` and pixel values are normalized.
  Args:
    image_path: Path to image.
    input_size: `int32` The dimension to resize the image. `(height, width)`.
    bbox: `float32` Normalized coordinates (0 to 1) in image where ocr needs to done. Shape is `(4)` like `[Xmin, Ymin, Xmax, Ymax]`
        if `bbox` is none, then no cropping is performed.
  Returns:
    `float32` numpy array of normalized image with shape `(input_size, 1)`.
  """
  input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  w, h = input_data.shape
  assert w == h, f'width and height of input image should be same, w = {w}, h = {h}'

  if bbox is None:
    bbox = [0, 0, input_data.shape[0], input_data.shape[1]]
  else:
    bbox = [int(b*w) for b in bbox]
  
  input_data = input_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
  input_data = cv2.resize(input_data, (input_size[1], input_size[0]))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data


class ReadingOCR:
  """ Text recognition of meeter reading. Given an input image, it uses OCR extraction to read the text in an image.

  Args:
    alphabets: List of output labels. `0123456789.`
    model_path: Path to tensorflow lite `ReadingOCR` model
    input_size: Input size of `ReadingOCR` model `(height, width)`
  """
  def __init__(
        self,
        alphabets,
        input_size=(31, 200),
        model_path="models/reading_ocr.tflite",
    ):

    self.model = load_tflite_model(model_path)
    self.alphabets = alphabets
    self.blank_index = len(alphabets)
    self.input_size = input_size

  def get_reading(self, image_path, bbox=None):
    """
    Extracts the reading from an image within the given area represented by the bounding box.
    Args:
      image_path: Input image path
      bbox: `float32` Normalized coordinates (0 to 1) in image where ocr needs to done. Shape is `(4)` like `[Xmin, Ymin, Xmax, Ymax]`
        if `bbox` is none, then OCR is done on whole image

      Returns:
        `string` Text contained in `bbox` area
    """
    input_data = prepare_input(image_path, self.input_size, bbox)

    # Get input and output tensors.
    input_details = self.model.get_input_details()
    output_details = self.model.get_output_details()

    self.model.set_tensor(input_details[0]['index'], input_data)
    self.model.invoke()

    output = self.model.get_tensor(output_details[0]['index'])

    text = "".join(self.alphabets[index] for index in output[0] if index not in [self.blank_index, -1])
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='tests/5.png')
    parser.add_argument('--model', type=str, default='models/reading_ocr.tflite')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
      print(f'{args.image} does not exist')
      sys.exit()

    alphabets = string.digits + string.ascii_lowercase + '.'
    rocr = ReadingOCR(alphabets=alphabets, model_path=args.model)

    text = rocr.get_reading(args.image, None)
    print(f'Extracted text: {text}')

if __name__=="__main__":
    main()