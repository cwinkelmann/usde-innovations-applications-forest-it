## TODO

### Setup

- [ ] figure out the environments
  - [x] `fit-megadetector` for P1–P2, data exploration, basic inference
  - [x] `fit-training` for P3–P8, full training pipeline with ultralytics, SAHI, timm, NOT GDAL/rasterio/megadetector (see below)
  - [x] `fit-herdnet` for active-learning, advanced geospatial/HerdNet work with GDAL/rasterio and animaloc (from git, not pip)

- [x] write installation instructions for each environment, including which practicals use which environment
- [ ] Describe how to run Label Studio via Docker for annotation practical
- [ ] Setup Jupyter Hub for remove training
[ ] There is this mention about the conversion of formats: Three common bounding box formats exist — you will convert between them in Practical 3:

## Camera Trap Detection
- [ ] Run the data inspection Notebook after the megadetector 

## Aerial Object Detection

- [ ] Aerial object detecion using using 
  - [ ] YOLO
  - [ ] HerdNet
  - [ ] HerdNet Class Number Modification


## Add Excercises
- [ ] discuss confidence reduction for small objects
- [ ] look into fp and fn for small objects

###  Introduction to my work of tiny object detection in aerial images in the Iguanas from Above Project
- [ ] https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb
- [ ] Herdnet


## Segmentation


## Training
- [ ] Train YOLO and HerdNet on the Eikelboom dataset, and compare their performance on tiny object detection in aerial images.


### Short presentation
- [ ] short slide deck to show the results of the tiny object detection in aerial images in the Iguanas from Above Project, and to introduce the problem of tiny object detection in aerial images more generally.