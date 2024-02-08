# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import cv2
import json
import requests
from tqdm import tqdm

sample_test_data = test_images[580:590]
sample_test_labels = test_labels[580:590]
sample_test_data_processed = np.expand_dims(sample_test_data / 255., axis=3)

data = json.dumps({"signature_name": "serving_default", 
                   "instances": sample_test_data_processed.tolist()})

HEADERS = {'content-type': 'application/json'}
MODEL1_API_URL = 'http://localhost:8501/v1/models/fashion_model_serving/versions/1:predict'

json_response = requests.post(MODEL1_API_URL, data=data, headers=HEADERS)
predictions = json.loads(json_response.text)['predictions']
predictions = np.argmax(np.array(predictions), axis=1)
prediction_labels = [class_names[p] for p in predictions]

fig, ax = plt.subplots(2, 5, figsize=(14, 6))
for idx, img in enumerate(sample_test_data):
    rowidx = idx // 5
    colidx = idx % 5
    ax[rowidx, colidx].imshow(img)
    ax[rowidx, colidx].set_title('Actual: {}\nPredicted: {}'.format(class_names[sample_test_labels[idx]], 
                                                                    prediction_labels[idx]), fontsize=10)
