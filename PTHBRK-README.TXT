This is forked from https://github.com/ultralytics/yolov3.

The only enhancement is a helper script that enables downloading 
and readying datasets without having to install hefty libraries like
PyTorch/Torchvision. 

Useful when the dataset is to be stored in a different
machine than the Colab server where the main yolo objection detection will run. 

The code to do this downloading is already present in utils/general.py and
data/*.yaml. This fork just refactors them into a standalone helper script.
