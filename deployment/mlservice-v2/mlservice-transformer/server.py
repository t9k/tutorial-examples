import json
from t9k import mlservice

import argparse
import io
import numpy as np
from PIL import Image
from typing import Dict

# inherit parser to get command-line-args help
# feel free to add your command line args
par = argparse.ArgumentParser(parents=[mlservice.option.parser])
args, _ = par.parse_known_args()

def image_transform(instance):
    image = Image.open(io.BytesIO(instance))
    a = np.asarray(image)
    a = a / 255.0
    return a.reshape(28, 28, 1).tolist()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def get_prediction(prediction: list)->str:
    max_value = max(prediction)
    max_index = prediction.index(max_value)
    return class_names[max_index]

class Transformer(mlservice.MLTransformer):
    def preProcess(self, path: str, requestData: bytes, headers: Dict[str, str]) -> any:
        return json.dumps({'instances': [image_transform(requestData)]})

    def postProcess(self, path: str, statusCode: int, responseContent: bytes, headers: Dict[str, str]) -> bytes:
        data = responseContent.decode('utf-8')
        jsonData = json.loads(data)
        jsonStr = json.dumps({'predictions': [get_prediction(predict) for predict in jsonData['predictions']]})
        return jsonStr.encode('utf-8')
    
if __name__ == "__main__":
    transformer = Transformer()
    server = mlservice.MLServer()
    server.start(transformer=transformer)