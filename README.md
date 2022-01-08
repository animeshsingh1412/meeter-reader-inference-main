## Extraction of readings from electricity meeter
Extraction of electricity meeter reading using machine learning using
- Object detection (Mobilenst SSD)
- Text detection (EAST)
- Text recognition (CRNN)

>Input images shape needs to have same height and width

### Install

#### Using Docker (recommended)
Install docker, https://docs.docker.com/engine/install/

```console
docker run -p 8501:8501 renjithks/meeter-reader-inference:latest
```
Then go to, http://localhost:8501

#### Using pip

Tested with `python` version `3.8`, although, any version higher than `3.7` should also work.

To install the dependencies
```console
pip install -r requirements.txt
```

Install lanms.
- Linux
  ```console
  cd lanms
  cp Makefile.nix Makefile
  make
  ```
- Mac
  ```console
  cd lanms
  cp Makefile.mac Makefile
  make
  ```
- Windows
  ```console
    cd lanms
    cp Makefile.win Makefile
    make
  ```
  [How to compile lanms on Windows?](https://github.com/argman/EAST/issues/120)
### Evaluation
To run the inferece on a single image.
```console
python run.py --image tests/2.png
```
![tests/2.png](output/2.png "") ![tests/3.png](output/3.png "")
![tests/4.png](output/4.png "") ![tests/6.jpeg](output/6.jpeg "")

To run inference on images in a directory.
```console
python run.py --input_folder tests
```
Predictions are saved to `output` folder by default.