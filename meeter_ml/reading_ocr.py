import cv2
import numpy as np
import string
import tensorflow as tf
import argparse
import os.path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='tests/5.png')
parser.add_argument('--model', type=str, default='models/reading_ocr.tflite')

def load_tflite_model(model_path):
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter

def prepare_input(image_path, input_size, bbox=None):
  input_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  if bbox is None:
    bbox = [0, 0, input_data.shape[0], input_data.shape[1]]
  
  input_data = input_data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
  input_data = cv2.resize(input_data, (input_size[1], input_size[0]))
  input_data = input_data[np.newaxis]
  input_data = np.expand_dims(input_data, 3)
  input_data = input_data.astype('float32')/255
  return input_data


class ReadingOCR:
  """ Text recognition of meeter reading

  Args:
    alphabets: list of output labels
    model_path: path to tensorflow lite model
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
    Args:
      image_path: input image path
      bbox: area in image where ocr needs to done, if bbox is none, then OCR is done on whole image

      return text contained in bbox area
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