# Object-Detector-App

A real-time object recognition application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/object_detection) and [OpenCV](http://opencv.org/).

## Getting Started
1. `conda env create -f environment.yml`
2. `python object_detection_app.py`

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.1](http://opencv.org/)

## Notes
- OpenCV might crash on OSX after a while. See open issue and solution [here](https://github.com/opencv/opencv/issues/5874)
- Moving the `.read()` part of the video stream in a subprocess did not work but only to a separate thread

## Copyright

See [LICENSE](LICENSE) for details.
Copyright (c) 2017 [Dat Tran](http://www.dat-tran.com/).
