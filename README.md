# Sea Surface Temperature (SST) Bias Correction using Neural Networks

Accurate Sea Surface Temperature (SST) measurements are fundamental to understanding the Earth's climate system, as SST directly influences global weather patterns, ocean currents, and the health of marine ecosystems. Research has shown that even slight variations in SST can significantly impact atmospheric circulation, driving phenomena such as El Niño and La Niña, which, in turn, affect global climate patterns. Reliable SST data is also crucial for improving climate models, which rely on accurate oceanic temperatures to forecast future climate scenarios. This project focuses on predicting the SST bias — the difference between SST values derived from satellite simulations (`TSkin_sim`) and ground-based observations (`TSkin_obs`). Addressing this bias is essential for enhancing the accuracy of climate monitoring and prediction efforts, ultimately contributing to a better understanding of climate change dynamics.


### Why Correct SST Biases?

Satellite observations are a primary source of global SST data due to their wide spatial coverage and frequent sampling. However, biases often exist in satellite-derived SST measurements due to factors such as
- **Instrument Calibration**: Differences in sensor sensitivity can lead to systematic errors in temperature readings.
- **Atmospheric Effects**: The presence of clouds, water vapor, aerosols, and other atmospheric constituents can interfere with the satellite’s infrared or microwave readings.
- **Surface Conditions**: Variability in surface types (e.g., sea ice, land proximity) can affect the accuracy of SST estimates.
  
These biases can lead to inaccuracies in climate models and weather forecasting. Therefore, developing reliable correction methods is essential to ensure the integrity of SST data for research and operational applications.

### Machine Learning for Bias Correction

Traditional methods for correcting SST biases often rely on statistical models or physical algorithms that require extensive calibration and validation. Machine learning, particularly deep learning, has shown promise in this domain due to its ability to capture complex, non-linear relationships between multiple variables.

In this project, we utilize a neural network to predict and correct SST biases. This approach leverages a dataset that includes both simulation and observation data, allowing the model to learn the underlying patterns associated with bias formation. By using input features like brightness temperatures, zenith angles, and atmospheric variables, the model aims to deliver more accurate SST estimates, enhancing the utility of satellite-derived climate data.

The project aligns with NOAA's mission to provide high-quality environmental information that supports weather prediction, climate monitoring, and ecosystem protection.

## Dependencies
This project is implemented using Python 3.x and requires the following packages:
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `netCDF4`
- `scikit-learn`

# Getting Started
## Installation
To install required packages, you can use the following command:
```
pip install tensorflow numpy pandas matplotlib netCDF4 scikit-learn
```
Make sure you have access to the netCDF files provided by NOAA. These files contain both satellite simulation data (`mirs_img_*.nc`) and observation data (`obs_img_*.nc`). Each file contains variables related to SST and atmospheric conditions, e.g. `Latitude`/`Longitude`, `Sfc_type`, `TSkin`, `TPW`, `CLW`, `LZ_angle`, `Emis`, `YM`. 

## Data Preprocessing
First, the script reads simulation and observation data from netCDF files and extracts key variables from the data. Then, the data is filtered to include only ocean samples (i.e. `Sfc_type = 0) to ensure the model focuses on oceanic SST predictions. Relevant simulation and observation data are combined and cleaned to remove missing or invalid values (e.g. SST values less than -900). The cleaned dataset is split evenly into training and testing sets. Lastly, the data is normalized based on training set statistics to standardize the input features for improved model performance.

## Model Architecture
The model is a feedforward neural network (FNN) built using Tensorflow/Keras with the following layers:
- **Input Layer**: contains 25 input features (latitude, longitude, brightness temperatures from 22 channels, local zenith angle)
- **Hidden Layers**: 2 dense layers with 200 nodes each using ReLU activation
- **Output Layer**: a single node predicting SST bias

## Training
The model is compiled using RMSprop optimizer with Mean Squared Error (MSE) as the loss function. Training runs for 1000 epochs with early stopping to prevent overfitting. Performance is assessed using Mean Absolute Error (MAE) and MSE on both training and validation sets. The `plot_history` function visualizes learning curves, tracking training and validation errors over epochs.

## Results
The output directory includes the trained neural network model (`.h5` format) and MSE/MAE plots for analysis.  
**Note**: Ensure adequate memory and computing resources to manage the large dataset.

## Acknowledgement
Special thanks to the NOAA MiRS team for their insights and support in advancing this project!

