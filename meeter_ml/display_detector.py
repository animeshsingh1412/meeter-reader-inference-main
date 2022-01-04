import sys
import logging
import tensorflow as tf

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def load_tflite_model(model_path, input_size):
  """
  Load the tf-lite model into memory.
  """

  logging.info(f'Loading tflite model: {model_path}')

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  # Note that the first frame will trigger tracing of the tf.function, which will
  # take some time, after which inference should be fast.
  # Run model through a dummy image

  preprocessed_image = tf.zeros([1, input_size, input_size, 3])
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  interpreter.set_tensor(input_details[0]['index'], preprocessed_image.numpy())
  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[1]['index'])
  classes = interpreter.get_tensor(output_details[3]['index'])
  scores = interpreter.get_tensor(output_details[0]['index'])

  logging.info(f'Successfully loaded tflite model: {model_path}') 

  return interpreter

class DisplayDetector:
  """ Detect the display of meeter using Mobilenet SSD object detection

  Args:
    model: path to tensorflow lite model
  """
  def __init__(
        self,
        model_path="model/display_detection.tflite",
        input_size=320
        
    ):

    self.model = load_tflite_model(model_path, input_size)

  def detect(self, image):
    """
    Detect the position of reading in an image.
    Returns the poly above threhold
    
    Args:
      image: is a numpy array
    """

    input_details = self.model.get_input_details()
    output_details = self.model.get_output_details()

    image = (2.0/255.0) * tf.convert_to_tensor(image, dtype=tf.float32) - 1.0
    image_tensor_batch = tf.expand_dims(image, axis=0) # batch_size = 1
    #processed_tensors = tf.image.per_image_standardization(image_tensor_batch)
    #

    self.model.set_tensor(input_details[0]['index'], image_tensor_batch)
    self.model.invoke()

    boxes = self.model.get_tensor(output_details[1]['index'])
    classes = self.model.get_tensor(output_details[3]['index'])
    scores = self.model.get_tensor(output_details[0]['index'])

    # Get box with highest confidence score
    idx = tf.math.argmax(scores[0])
    score = scores[0][idx]
    class_id = classes[0][idx]

    w = image.shape[0]  # square image w=h
    box =  tf.cast(boxes[0][idx]*w, tf.int32).numpy()
    box = [w if x > w else x for x in box] # crop box if its overflowing
    box = [0 if x < 0 else x for x in box]

    #convert [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    box = [box[1], box[0], box[3], box[2]]

    return box, class_id, score

