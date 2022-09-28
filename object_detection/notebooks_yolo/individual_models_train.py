import os
for variable in ['bus', 'car', 'motorcycle', 'parking_meter', 'stop_sign', 'taxi', 'traffic_light', 'traffic_sign', 'vehicle_registration_plate']:
    os.system('python /home/jupyter/yolov5/train.py \
            --img 1280 \
            --batch 4 \
            --epochs 10 \
            --data /home/jupyter/{variable}/YOLO_store.yaml \
            --weights yolov5x6.pt  \
            --project /home/jupyter/{variable} \
            --name output \
            --patience 5 \
            --label-smoothing 0.1 \
            --seed 42'.format(variable=variable))