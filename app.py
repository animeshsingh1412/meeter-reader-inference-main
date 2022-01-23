import streamlit as st
import cv2
import numpy as np
import tempfile
import glob
import shutil
from pathlib import Path

from config import settings

from meeter_ml import display_detector as dd
from meeter_ml import reading_detector as rd
from meeter_ml import reading_ocr as rocr
from run import inference

st.title("Electricity meeter reading inference")

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

# store locally
input_folder = tempfile.mkdtemp()

uploaded_files = st.file_uploader("Choose an image with same widht and height", accept_multiple_files=True, type=['jpeg', 'jpg', 'png'])
for uploaded_file in uploaded_files:
  # Convert the file to an opencv image.
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  opencv_image = cv2.imdecode(file_bytes, 1)

  cv2.imwrite(input_folder + '/' + uploaded_file.name, opencv_image)


for i, file in enumerate(glob.glob(input_folder +"/*.*")):
  if any([ext in file.lower() for ext in ['.jpeg', '.jpg', '.png']]):
    
    try:
      reading, img_out = inference(
        file,
        display_detector,
        reading_detector,
        reading_recogniser
    ) 
      if reading is not None:
        reading = reading.split(".")[0]
    except AssertionError as e:
      st.caption(f'{Path(file).name}')
      st.error(f'Error {Path(file).name}: {e}')
      #col.image(img_out, channels="BGR", use_column_width=True)
      st.markdown("""---""")
      continue

  if img_out is not None:
    img_out = cv2.resize(img_out, (320, 320))
  else:
    img_out = cv2.resize(cv2.imread(file), (320, 320))
    reading = None

  st.caption(f'{Path(file).name}')
  st.subheader(f'{reading}')
  st.image(img_out, channels="BGR")
  st.markdown("""---""")

shutil.rmtree(input_folder)
