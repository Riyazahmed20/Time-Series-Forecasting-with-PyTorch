## Time Series Forecasting with PyTorch

## Objective
To build and train a deep learning model using PyTorch for univariate time series forecasting. This project aims to demonstrate how LSTM networks can be used to predict future values based on historical sequential data.

## Description
Time series forecasting is widely used in domains like finance, energy, and climate modeling. This notebook walks through:
- Generating or loading time series data
- Creating input-output sequences
- Building an LSTM-based forecasting model using PyTorch
- Training the model and evaluating performance
- Visualizing the prediction vs actual values

The model predicts future time steps using past observations and learns temporal dependencies using recurrent layers (LSTM).

## Training Loss Curve

The training loop monitors and visualizes the Mean Squared Error (MSE) loss over epochs, helping to understand convergence behavior.

```python
import matplotlib.pyplot as plt

plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()

```

## How to Run
Clone Repository:
git clone https://github.com/yourusername/Time-Series-Forecasting-with-PyTorch.git
cd Time-Series-Forecasting-with-PyTorch

## Dependencies
Make sure you have the following installed:

- Python 3.8 or higher  
- PyTorch  
- NumPy  
- Matplotlib  
- scikit-learn  

## Install all dependencies using pip:
```bash
pip install torch numpy matplotlib scikit-learn

```

