import os
from ultralytics import YOLO

from IPython.display import display, Image
from IPython import display
from roboflow import Roboflow
import ultralytics
import subprocess


HOME = os.getcwd()
os.chdir("/root/supervision/examples/traffic_analysis/datasets")



rf = Roboflow(api_key="72lnKOD8oM1rj5Q6UsPk")
project = rf.workspace("sma-nx6wx").project("transit-model")
dataset = project.version(4).download("yolov8")

os.chdir(HOME)

subprocess.run(["yolo", "task=detect", "mode=train", "model=yolov8s.pt", f"data={dataset.location}/data.yaml", "epochs=50", "imgsz=800", "plots=True"])


'''
!mkdir {HOME}/datasets
%cd {HOME}/datasets

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="72lnKOD8oM1rj5Q6UsPk")
project = rf.workspace("sma-nx6wx").project("transit-model")
dataset = project.version(4).download("yolov8")
'''