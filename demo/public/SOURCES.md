# Data and model sources

The 30 displayed traffic-sign crops originate from the German Traffic Sign Recognition Benchmark (GTSRB), not from newly collected camera footage.

- Official authors' archive: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
- Inputs: GTSRB_Final_Test_Images.zip and GTSRB_Final_Test_GT.zip. Only the 30 selected, padded and resized ROI crops and their processed derivatives are included here.
- Research source and reproduction instructions: https://github.com/kttt294/GTSRB-Low-Light-Preprocessing
- Executed notebook: https://www.kaggle.com/code/kieuthithutrang/gtsrb-low-light-enhancement-for-traffic-signs

All predictions were computed offline with the group's frozen ResNet-18 checkpoint, training seed 11. Low light is simulated, with noise seed 101; the selected enhancement pipeline is Gaussian filtering followed by gamma correction. This website displays precomputed results and does not run online inference.

The complete file hashes, checkpoint identifier, source-code hashes and result provenance are provided in data/provenance.json.
