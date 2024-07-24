from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from typing import List

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model from checkpoint
def load_model(checkpoint_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Define input data model
class InputData(BaseModel):
    input: List[float]

# Create FastAPI app
app = FastAPI()

# Load the model checkpoint
try:
    model = load_model('simple_nn_checkpoint.pth')
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.get("/model_check")
def model_check():
    if model_loaded:
        return {"status": "Model loaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Model failed to load")

@app.post("/predict")
def predict(data: InputData):
    if len(data.input) != 10:
        raise HTTPException(status_code=400, detail="Input must be a list of 10 floats.")
    
    input_tensor = torch.tensor(data.input).float().unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    
    return {"output": output.squeeze().item()}

@app.get("/inference_check")
def inference_check():
    try:
        test_input = torch.randn(1, 10)
        with torch.no_grad():
            test_output = model(test_input)
        return {"status": "Inference successful", "output_sample": test_output.squeeze().item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)