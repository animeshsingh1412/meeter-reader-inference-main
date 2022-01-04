import argparse
import cv2
import numpy as np
from tensorflow.python.ops.gen_math_ops import imag

from meeter_ml import display_detector as dd
from meeter_ml import reading_detector as rd
from tools import load_image_into_numpy_array, draw_bb
from config import settings

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
    

def detect_display(image_path, display_detector):
  image = load_image_into_numpy_array(image_path)
  input_size = settings.display_detection.input_size
  image_resized = cv2.resize(image, (input_size, input_size), cv2.INTER_CUBIC)
  box, class_id, score = display_detector.detect(image_resized)
  return box, class_id, score

def detect_reading(image_path, reading_detector):
  sorted_polys, img_out = reading_detector.detect(image_path)
  return sorted_polys, img_out

def inference(image_path, display_detector, reading_detector):
  image = load_image_into_numpy_array(image_path)

  w, h, _ = image.shape

  assert w == h, f'width and height of input image shoud be same, got width {w} and height {h}'
  assert w % 32 == 0 and h % 32 == 0  , f'width and height of input image shoud multiple of 32, got width {w} and height {h}'

  box_display, class_id, score = detect_display(image_path, display_detector)
  score_display = int(score*100)

  text_det_poly, img_text_det_reading = detect_reading(image_path, reading_detector)
  best_text_poly = select_best_text_poly(box_display, text_det_poly)

  best_text_box =  [
    min(best_text_poly[:,:,0])[0],
    min(best_text_poly[:,:,1])[0],
    max(best_text_poly[:,:,0])[0],
    max(best_text_poly[:,:,1])[0]
    ]

  return image, box_display, score_display, best_text_box

def main():
  """
  Run the inference on a single image
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", default="tests/1.png", help="Path to image")
  args = parser.parse_args()

  display_detector = dd.DisplayDetector(
    settings.display_detection.model_path,
    settings.display_detection.input_size
  )
  reading_detector = rd.ReadingDetector(settings.reading_detection.model_path)

  image, box_display, score_display, best_text_box = inference(
      args.image,
      display_detector,
      reading_detector
    )

  img_out = draw_bb(
    image=image,
    rect=best_text_box,
    text=f'reading',
    color=(255, 255, 0))

  img_out = draw_bb(
    image=img_out,
    rect=box_display,
    text=f'display {score_display}',
    color=(0, 255, 0))

  cv2.imshow('image', img_out)
  cv2.waitKey(0)

if __name__=="__main__":
  main()