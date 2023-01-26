# Face_Detect_RetinaFace
Project to use RetinaFace pretrained models to detect faces


Two models are provided that have been trained using the [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface) protocol. The `Resnet50_Final_model.pth` provides the better performance than the `mobilenet0.25_Final_model.pth`, but the model is larger and so is hosted off github.  [Here's a link to the Resnet model](https://drive.google.com/file/d/19A6wrCTJm-v2c606JIfDSC0uLUL5ovSP/view?usp=share_link). Once downloaded, put the model in the `models` directory.  If you do not download this model, the model used by the Face_Detect code will default to the `mobilenet0.25_Final_model.pth`.

## Set up an environment via conda

The easiest way to consume this project is to set up a conda environment and run the code within it. If you do not have Anaconda or one of its variants installed, [installation instructions for Anaconda can be found here.](https://docs.anaconda.com/anaconda/install/) 

Once you have Anaconda installed for your operating system, the following command will set up a conda environment named 'faceTest' and install the dependencies in that environment required to run this project :

```
conda create --name faceTest python==3.8.5 pytorch torchvision cpuonly -c pytorch -y && conda activate faceTest2 && pip install opencv-python
```

When the command is finished, you will be in the 'faceTest' environment.  Navigate to the directory where you copied the Face_Detect_RetinaFace project, and use the following commands to execute the code :

To run the face detection on a single image (from the repo root directory) :

```
python script.py <relative image path/name>.<jpg or png>

```
So, for example, using the current repository's layout, with source images in the `images` directory