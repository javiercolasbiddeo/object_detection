import torch, torchvision

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import random
from collections import Counter
import json
from matplotlib import pyplot as plt
import pickle
import time


# import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def categories_orden(json_file):
    orden_categorias = []
    for i in json_file['categories']:
        orden_categorias.append(i['name'])
        
    return orden_categorias

def class_frequency(json_file):
    
    categorias = []
    for i in json_file['annotations']:
        categorias.append(i['category_id'])

    print(f'Number of categories: {len(set(categorias))}')
    print(f'frequency: {Counter(categorias)}')

    

    
def train_individual_models(variable):
    
    f = open(f'/home/jupyter/notebook-javi/object_detection/detectron_2/{variable}/labels_train') # dale el path del json
    json_file_train = json.load(f)
    
    f = open(f'/home/jupyter/notebook-javi/object_detection/detectron_2/{variable}/labels_test') # dale el path del json
    json_file_test = json.load(f)
    
    register_coco_instances(f"data_train_{variable}",
                        {},
                        f'/home/jupyter/notebook-javi/object_detection/detectron_2/{variable}/labels_train',
                        '/home/jupyter/main_dataset/train/data') 
    
    register_coco_instances(f"data_test_{variable}",
                        {},
                        f'/home/jupyter/notebook-javi/object_detection/detectron_2/{variable}/labels_est',
                        '/home/jupyter/main_dataset/test/data') 
    
    orden_categorias = categories_orden(json_file_train)
    
    with open(f'{variable}/detectron2_order_categories.pckl', 'wb') as handle:
        pickle.dump(orden_categorias, handle)
        
        
        
    # Fijamos los hiperparametros.

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) # Habrá que ver los modelos que existen y cuales se adaptan mejor a la tarea que queremos
    cfg.DATASETS.TRAIN = (f"data_train_{variable}",)
    cfg.DATASETS.TEST = () # para controlar el overfitting sería ideal separar en train y val, como es una prueba pasamos.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.SOLVER.IMS_PER_BATCH = 2 # El número de batch es a elección nuestra. Contra más iteraciones más numero de batch es lo ideal
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 12500
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(json_file_train['categories']) # Fijamos el numero de categorias. En nuestro ejemplo hay 601, pero realmente solo estamos usando fotos con 91 categorías
    cfg.OUTPUT_DIR = f'/home/jupyter/notebook-javi/object_detection/detectron_2/{variable}/output'
    cfg.TEST.EVAL_PERIOD = 500
    
    
    
    with open(f'{variable}/pipeline.pckl', 'wb') as handle:
        pickle.dump(cfg, handle)
        
        
    print('Frecuencia de clases Train')
    class_frequency(json_file_train)
    
    print('Frecuencia de clases Test')
    class_frequency(json_file_test)
    
    
    start_time = time.time()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False) # revisar esto
    trainer.train()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
for categoria in ['ambulance', 'bicycle', 'bus', 'car', 'motorcycle', 'parking_meter', 'stop_sign', 'taxi', 'traffic_light', 'traffic_sign', 'vehicle_registration_plate']:
    train_individual_models(categoria)