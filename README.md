Occlusion-Net\_extended: Extending [[Occlusion-Net](https://github.com/dineshreddy91/Occlusion\_Net)] for video analysis context.
======================
This repository extends original [[Occlusion-Net](https://github.com/dineshreddy91/Occlusion\_Net)] repository in order to run its object detection pipeline on traffic survelliance videos collected at the [[Chair of Integrated Transport Planning and Traffic Engineering](https://tu-dresden.de/bu/verkehr/ivs/ivst/studium?set_language=en)].

The Occlusion-net\_extended extends Occlusion-Net as follows:

1) Video files are now accepted as inputs and Occlusion-Net outputs the detected video file.

2) In addition to detected wireframe, the processed video files also have overlaid (drawn) bounding boxes (to aid qualitative evaluation).

3) Frames which are skipped (not processed) by Occlusion-Net are stored separately as individual images (and not videos).

4) The .json output for each object now contains predicted keypoint locations (earlier only BB coordinates) and their confidence scores.

5) The bounding box coordinates of each object now match with their corresponding keypoints (earlier BB with ID=1 had keypoints corresponding to BB with ID=2).

6) Additional python modules are available for extracting frames from videos to process them as an image directory and for combining the processed frames into a video. These are done in parallel.


## Installation and Dataset setup
Refer to original\_README.md for setting up Occlusion-Net and for information on dataset and training scripts (if retraining).

Alternatively, pre-trained weights can also be downloaded. [[Google Drive](https://drive.google.com/open?id=1EUmhzeuMUnv5whv0ZmmOHTbtUiWdeDly)]



### Running with video detection pipeline

1) Clone the Occlusion-Net\_extended repository.

2) Build docker-image

```
nvidia-docker build -t occlusion\_net.

```

3) Move the video to be processed in Occlusion\_Net/demo folder.

4) Run detection pipeline on video. For eg. video name = test.mp4

```
sudo sh test.sh occlusion\_net ./demo/test.mp4
```
5) Processed video file and missing frames are stored in Occlusion\_Net/log folder.




