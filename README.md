# Face_Detect_RetinaFace
Project to use RetinaFace pretrained models to detect faces.

To access this project, you can clone this repository, or [download an archive of the code here](https://github.com/jturner65/Face_Detect_RetinaFace/archive/refs/heads/main.zip) and unzip in a desired directory.

Two models are provided that have been trained using the [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) protocol. The `Resnet50_Final_model.pth` provides the better performance than the `mobilenet0.25_Final_model.pth`, but the model is larger and so is hosted off github.  [Here's a link to the Resnet model](https://drive.google.com/file/d/19A6wrCTJm-v2c606JIfDSC0uLUL5ovSP/view?usp=share_link). Once downloaded, put the model in the `models` directory.  If you do not download this model, the model used by the Face_Detect code will default to the `mobilenet0.25_Final_model.pth`.

## Set up an environment via conda

The easiest way to consume this project is to set up a conda environment and run the code within it. If you do not have Anaconda or one of its variants installed, [installation instructions for Anaconda can be found here.](https://docs.anaconda.com/anaconda/install/) 

Once you have Anaconda installed for your operating system, the following command will set up a conda environment named 'faceTest' and install the dependencies in that environment required to run this project :

```
conda create --name faceTest python==3.8.5 pytorch torchvision cpuonly -c pytorch -y && conda activate faceTest2 && pip install opencv-python
```

## Executing the code

When the above command is finished, you will be in the 'faceTest' environment.  Navigate to the directory where you copied the Face_Detect_RetinaFace project, and use the following commands to execute the code :

To run the face detection on a single image (from the repo root directory) :

```
python script.py <relative image path/name>.<jpg or png>
```
So, for example, using the current repository's layout, with source images in the `images` directory, you would execute the script on `0_faces_4.jpg` using the following :

```
python script.py images/0_faces_4.jpg 
```

If instead, you wish to execute the script on an entire directory, such as all images in the `images/` directory, you would use : 

```
python script.py images
```
The script also supports the following command-line parameters : 

```
usage: script.py [-h] [-e] [-n {resnet50,mobile025}] [-s] [-v] file_or_dir_name

Count faces in images

positional arguments:
  file_or_dir_name      Either the name of an image to find and count faces in, or the name of a directory containing a list of such images. Should be .jpg or .png extension.

optional arguments:
  -h, --help            show this help message and exit
  -e, --eval            Evaluate results for ranges of confidence and non-maximal threshold values [0-1) 
                        and save the results as csv files, and, if matplotlib is installed, via 3d plots.
  -n {resnet50,mobile025}, --network {resnet50,mobile025}
                        Backbone network to use for detection.
  -s, --save_image      Save annotated images showing detection results.
  -v, --verbose_msgs    Print verbose results to the console.
```


## Evaluating the models

I chose RetinaFace over the other 2 possibilities because it seemed to be the most recent, was the easiest for me to install and use, and also seemed to require the least dependencies to consume its model (only requiring pytorch and opencv). I was also intrigued by its premise and was genuinely curious about how it worked, regarding the facial feature detection.

The evaluation process provided shows how the choice of confidence and non-maximal suppression thresholds impacts the # of incorrect mapping counts. Unfortunately, without some kind of geometric(bbox) oracle/ground-truth annotations to validate against, evaluating the nature of proposals (i.e. # of false-positive/false-negative mappings) is difficult, and requires checking manually.  

Also, it would appear that some of the image-title annotations are inaccurate.  For example, 2_faces_3.jpg clearly has 3 faces visible - the 2 obvious faces looking at the camera, and one off to the left of center in profile.  Both models were able to find this third face. Another example is 9_faces_1.jpg, which is annotated to contain 9 faces, but upon visual inspection only has 8 that are distinguishable, with at least half the face showing.


