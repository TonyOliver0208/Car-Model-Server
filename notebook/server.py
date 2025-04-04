# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import torchvision.models as models
# from PIL import Image
# import io
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_methods=["*"],  
#     allow_headers=["*"],  
# )

# model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 196)  

# def find_classes(dir):
#     classes = os.listdir(dir)
#     classes.sort()  
#     return classes

# local_path = '../data/car_data'
# classes = find_classes(os.path.join(local_path, 'train'))
# model.fc = nn.Linear(num_ftrs, len(classes))  

# model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
# model.eval() 


# transform = transforms.Compose([
#     transforms.Resize((400, 400)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = Image.open(io.BytesIO(await file.read()))
#         image = transform(image).unsqueeze(0)  

#         with torch.no_grad():
#             output = model(image)
#             confidence, predicted_class = torch.max(output.data, 1)

#         return {"car": classes[predicted_class.item()], "confidence": float(confidence.item())}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load the trained model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)  
num_ftrs = model.fc.in_features

# Load class labels from the file
with open('classes.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Update the model's final layer to match the number of classes
model.fc = nn.Linear(num_ftrs, len(classes))  

# Load the model weights
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval() 

# Define the same transformations used during training
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(await file.read()))
        image = transform(image).unsqueeze(0)  

        # Perform inference
        with torch.no_grad():
            output = model(image)
            confidence, predicted_class = torch.max(output.data, 1)

        # Return predicted car name and confidence score
        return {"car": classes[predicted_class.item()], "confidence": float(confidence.item())}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
