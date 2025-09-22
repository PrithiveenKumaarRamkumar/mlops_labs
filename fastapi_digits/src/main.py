'''
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

#UPDATE 1

'''
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List
from predict import predict_data

app = FastAPI()

class DigitData(BaseModel):
    pixels: List[float]  # Flattened 8x8 = 64 values

class DigitResponse(BaseModel):
    response: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=DigitResponse)
async def predict_digit(digit_features: DigitData):
    try:
        if len(digit_features.pixels) != 64:
            raise HTTPException(status_code=400, detail="Input must be 64 values representing an 8x8 image")

        features = [digit_features.pixels]  # Keep in 2D shape
        prediction = predict_data(features)
        return DigitResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

#UPDATE 2

from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import List
from predict import predict_data
from sklearn.datasets import load_digits
import numpy as np
import random

app = FastAPI()

class DigitData(BaseModel):
    features: List[float]   # 64 length list for an 8x8 digit

class DigitResponse(BaseModel):
    response: int

class SampleResponse(BaseModel):
    features: List[float]
    predicted: int
    true_label: int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=DigitResponse)
async def predict_digit(digit_data: DigitData):
    try:
        if len(digit_data.features) != 64:
            raise HTTPException(
                status_code=400,
                detail="Input must be a list of 64 features (8x8 flattened image)."
            )

        features = [digit_data.features]  # wrap inside batch
        prediction = predict_data(features)
        return DigitResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sample", response_model=SampleResponse)
async def get_sample():
    """
    Return a random sample from the digits dataset along with model prediction.
    """
    digits = load_digits()
    idx = random.randint(0, len(digits.data) - 1)

    features = digits.data[idx].tolist()
    true_label = int(digits.target[idx])

    prediction = predict_data([features])

    return SampleResponse(
        features=features,
        predicted=int(prediction[0]),
        true_label=true_label
    )


    
