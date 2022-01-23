import cv2
import time
import os
import numpy as np
import tensorflow as tf
import argparse
import lanms

__pdoc__ = {
    'main': False
}

def load_tflite_model(model_path):
  """
  Load the tf-lite model into memory.
  Args:
    model_path: Path to tensorflow lite model for `ReadingDetector`
  Returns:
    model: Interpreter for tensorflow lite `ReadingDetector` model
  """

  model = tf.lite.Interpreter(model_path=str(model_path))
  model.allocate_tensors()

  return model

def resize_image(im, max_side_len=2400):
    '''
    Resize image to a size multiple of 32 which is required by the network.
    
    ```text
    The input image is already a multiple of 32. So this function in effect does not need to resize the image.
    We still keep this method incase in future, we diside to use other input dimensions which is not multiple of 32.
    ```
    
    Args:
        im: `uint8` numpy array with shape `(img_height, img_width, 3)`. The image to resize. 
        max_side_len: Limit of max image size to avoid out of memory in gpu
    Returns:
        im: The resized image.
        ratio_h: Resize ratio in y direction. Always 1.0 in our case, since the input image dimension is a multiple of 32.
        ratio_w: Resize ratio in x direction. Always 1.0 in our case, since the input image dimension is a multiple of 32.
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    
    return im, (ratio_h, ratio_w)


def extract_box(score_map, geo_map, timer, score_map_thresh=0.7, box_thresh=0.1, nms_thres=0.2):
    '''
    Restore text boxes from score map and geo map. 
    Args:
        score_map: `float32` Pixel level score map to tell about the confidence level prediction of text in it. Shape is `(1, 80, 80, 1)`. Range of score is `0~1`.
        geo_map: `float32` Pixel level Rotated Boxes containing 5 values of which 4 are top and left coordinate, width, height and one is rotation angle in counterclockwise direction.
            Shape is `(1, 80, 80, 5)`. The last 5 defines the rotated box. `[Y_top, X_left, width, height, angle_rotated]`.
        timer: Timing for network, ex:`{'net': 0, 'restore': 0, 'nms': 0}`. Used for performace analysis.
        score_map_thresh: Threshhold for score map `0~1`. Any score maps less than this threshold is ignored.
        box_thresh: Threshhold for boxes `0~1`. Any boxes maps less than this threshold is ignored.
        nms_thres: Non-maximum suppression threshold `0~1`.
    Returns:
        boxes: `float32`. List of final polygons along with its score. Shape is `(M, 9)`. `M` is number of polygons. `[[X0, Y0, X1, Y1, X2, Y2, X3, Y3, score]]`.
        timer: Timing for network, ex:`{'net': 0, 'restore': 0, 'nms': 0}`. Used for performace analysis.
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]

    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    return boxes, timer

def sort_poly(p):
    """
    A utility function to make sure that coordinates of polygon are in the right order starting from top left to bottom left in clockwise direction.
    Args:
        p: `int32` Coordinates of rectangle. Shape is `(4, 2)` ex: `[[X0, Y0], [X1, Y1], [X2, Y2], [X3, Y3]]`.
    Returns:
        `int32` Sorted (clockwise) rectangle coordinates. Shape is `(4, 2)`.
    """
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def restore_rectangle(origin, geometry):
    """
    From origin of text boxes and its geometry, compute the cordinate of the resulting rectangle.
    Args:
        origin: `int32` Origin points where text boxes are located. Shape is `(N, 2)`. 2 is the cordinate of origin `[X, Y]`
        geometry: `float32` Geometry of the corresponding origin coordinates.
            Rotated Boxes containing 5 values of which 4 are top and left coordinate, width, height and one is rotation angle in counterclockwise in direction.
            Shape is `(N, 5)`.
            5 contains `[Y_top, X_left, width, height, angle_rotated]`.
    Returns: `float32` A list of rectangles of shape `(N, 4, 2)`. 
        N is the number of rectangles.
        4, 2 is the coordinates of each rectangle `[X0, Y0], [X1, Y1], [X2, Y2], [X3, Y3]`
    """

    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])

class ReadingDetector:
  """ Detect the reading in meeter using EAST text detection
  `ReadingDetector` detects the location of all texts in an image.

  Args:
    model_path: Path to `ReadingDetector` tensorflow lite model
    input_size: `int32` Input size of `ReadingDetector` tensorflow lite model. `(height, width)`

  Returns:
    ReadingDetector: An instance of `ReadingDetector` class
  """
  def __init__(
        self,
        model_path="model/reading_detection.tflite",
        input_size=(320, 320)
    ):

    self.model = load_tflite_model(model_path)
    self.input_size = input_size

  def detect(self, image):
    """
    Detects the location of texts in an image.
    Image is fed into the FCN and multiple channels of pixel-level text score map and geometry are generated.
    One of the predicted channels is a score map whose pixel values are in the range of `[0, 1]`.
    The second channels represent geometries that encloses the word from the view
    of each pixel. The score stands for the confidence of the geometry shape predicted at the same location.
    Thresholding is then applied to each predicted region, where the geometries whose scores are over the predefined 
    threshold is considered valid and saved for later nonmaximum-suppression. Results after NMS are considered
    the final output of the pipeline.

    Args:
        image: Path to input image

    Returns:
        text_polys : `int32` List of detected polygons around texts. 4D array of shape `(N, 4, 1, 2)`, where N is number of text boxes detected
            ```python
            [[[[ X0,  Y0]], [[X1,  Y1]], [[X2,  Y2]], [[X3,  Y3]]]]
            ```
        img_out: `uint8` numpy array with shape `(img_height, img_width, 3)`. Annotated image with boxes around texts. 
    """
    input_index = self.model.get_input_details()[0]["index"]
    output_index_1 = self.model.get_output_details()[0]["index"]
    output_index_2 = self.model.get_output_details()[1]["index"]

    im_fn_list = [image]
    for im_fn in im_fn_list:
        im = cv2.imread(im_fn)
        im = cv2.resize(im, self.input_size, cv2.INTER_CUBIC)[:, :, ::-1]
        start_time = time.time()
        im_resized, (ratio_h, ratio_w) = resize_image(im)
        
        timer = {'net': 0, 'restore': 0, 'nms': 0}
        start = time.time()
        self.model.set_tensor(input_index, im_resized[np.newaxis, :, :, :].astype(np.float32))
        self.model.invoke()
        score = self.model.get_tensor(output_index_1)
        geometry = self.model.get_tensor(output_index_2)
        timer['net'] = time.time() - start

        boxes, timer = extract_box(score_map=score, geo_map=geometry, timer=timer)
        #print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        #    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        #print('[timing] {}'.format(duration))

        sorted_polys = []
        if boxes is not None:
            for box in boxes:
                # to avoid submitting errors
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                # f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                #     box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                # ))
                box = np.clip(box, 0, max(im_resized.shape))
                cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                sorted_polys.append(box.astype(np.int32).reshape((-1, 1, 2)))
            
            return sorted_polys, im[:, :, ::-1]
        else:
            return None, None


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="tests/4.png", help="Path to image")
    parser.add_argument("--model", default="models/reading_detection.tflite", help="Path to tflite readin detection model")
    args = parser.parse_args()

    rd = ReadingDetector(args.model)
    sorted_polys, im = rd.detect(args.image)

    print('[polygons]')
    print(sorted_polys)
    cv2.imshow('image', im[:, :, ::-1])
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
