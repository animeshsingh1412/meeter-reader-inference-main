import argparse
import cv2
import glob
import numpy as np

from pathlib import Path
from meeter_ml import display_detector as dd
from meeter_ml import reading_detector as rd
from meeter_ml import reading_ocr as rocr
from tools import draw_text, load_image_into_numpy_array, draw_bb
from config import settings

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Path to image")
parser.add_argument("--input_folder", help="Path to input folder")
parser.add_argument("--output_folder", default="output", help="Path to output folder")
args = parser.parse_args()

def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def select_best_text_poly(meeter_bbox, text_plygons):
  """
  Given a bbox of meeter display and text detection polygons,
  compute the best possible candiate amoung the text detection polygons.
  To find the best text box, compute text box having best IOU with beeter display box.

  Args:
    meeter_bbox: Detected meeter display box rectangle
    text_plygons: List of detected polygons around text

    returns the text polygon that has maximum IOU with bbox
  """
  ious = []
  for poly in text_plygons:
    x_min = min(poly[:,:,0])[0]
    y_min = min(poly[:,:,1])[0]
    x_max = max(poly[:,:,0])[0]
    y_max = max(poly[:,:,1])[0]

    text_rect = [x_min, y_min, x_max, y_max]

    ious.append(calculate_iou(meeter_bbox, text_rect))

  max_iou_idx = np.argmax(ious)
  return text_plygons[max_iou_idx]

def ocr_reading(image_path, reading_recognisier, bbox):
  reading = reading_recognisier.get_reading(image_path, bbox)
  return reading
    
def detect_display(image_path, display_detector):
  image = load_image_into_numpy_array(image_path)
  input_size = settings.display_detection.input_size
  image_resized = cv2.resize(image, (input_size, input_size), cv2.INTER_CUBIC)
  box, class_id, score = display_detector.detect(image_resized)
  return box, class_id, score

def detect_reading(image_path, reading_detector):
  text_polys, img_out = reading_detector.detect(image_path)
  return text_polys, img_out

def inference(image_path, display_detector, reading_detector, reading_recogniser):
  image = cv2.imread(image_path)

  w, h, _ = image.shape

  assert w == h, f'width and height of input image shoud be same, got width {w} and height {h}'
  assert 320 == settings.display_detection.input_size == \
    settings.reading_detection.input_size, \
    f'input size of reading_detection and display_detection in settings should be 320'

  box_display, _, score = detect_display(image_path, display_detector)
  score_display = int(score*100)

  text_det_poly, _ = detect_reading(image_path, reading_detector)
  
  if not box_display or not text_det_poly:
    return None, None

  best_text_poly = select_best_text_poly(box_display, text_det_poly)

  best_text_box =  [
    min(best_text_poly[:,:,0])[0] - 5,
    min(best_text_poly[:,:,1])[0] - 5,
    max(best_text_poly[:,:,0])[0] + 5,
    max(best_text_poly[:,:,1])[0] + 5
    ]
  best_text_box = np.clip(best_text_box, 0, settings.reading_detection.input_size)
  best_text_box_normalised = [float(x)/float(settings.display_detection.input_size) for x in best_text_box]
  box_display_normalised = [float(x)/float(settings.reading_detection.input_size) for x in box_display]
  
  reading = ocr_reading(image_path, reading_recogniser, best_text_box_normalised)

  img_out = draw_bb(
    image=image,
    rect=best_text_box_normalised,
    text=f'reading',
    color=(255, 255, 0))

  img_out = draw_bb(
    image=img_out,
    rect=box_display_normalised,
    text=f'display {score_display}',
    color=(0, 255, 0))

  return reading, img_out

def main():
  """
  Run the inference on a single image or on images in a folder
  writes output to output folder
  """

  display_detector = dd.DisplayDetector(
      settings.display_detection.model_path,
      settings.display_detection.input_size
    )

  reading_detector = rd.ReadingDetector(
      settings.reading_detection.model_path,
      (settings.reading_detection.input_size, settings.reading_detection.input_size)
    )

  reading_recogniser =  rocr.ReadingOCR(
      settings.reading_ocr.alphabet,
      (settings.reading_ocr.input_size.y, settings.reading_ocr.input_size.x),
      settings.reading_ocr.model_path
    )

  if args.image:
    reading, img_out = inference(
        args.image,
        display_detector,
        reading_detector,
        reading_recogniser
      )

    print(f'Extracted reading: {str(reading)}')

    if img_out is not None:
      cv2.imshow(str(reading), img_out)
      cv2.waitKey(0)
    
  if args.input_folder:
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)

    for file in glob.glob(args.input_folder +"/*.*"):
      if any([ext in file.lower() for ext in ['.jpeg', '.jpg', '.png']]):
        try:
          reading, img_out = inference(
            file,
            display_detector,
            reading_detector,
            reading_recogniser
        )
        except AssertionError as e:
          print(f'Error {file}: {e}')
          continue

      print(f'Results {file}: {reading}')

      if img_out is not None:
        img_out = cv2.resize(img_out, (320, 320))
        img_out = draw_text(img_out, reading)
        cv2.imwrite(args.output_folder + '/' + Path(file).name, img_out)
      

if __name__=="__main__":
  main()