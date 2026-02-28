from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

import sys
from netCDF4 import Dataset
import glob
import os

np.set_printoptions(threshold=sys.maxsize)


# =============================================================================
# User settings
# =============================================================================

date     = "12days"
dir_in   = "/ccsopen/proj/gen175/mirs/data/daily_data/"
dir_obs  = dir_in
out_dir  = "/gpfs/wolf/gen175/proj-shared/mirs/shu/4GPU/"

entire_model_name  = out_dir + "Entire_Model_node200_" + date + ".h5"
model_json_name    = out_dir + "model_node200_"         + date + ".json"
model_weights_name = out_dir + "model_node200_"         + date + ".h5"
history_name       = out_dir + "history_model_node200_" + date + ".csv"

nfov  = 96
nchan = 22


# =============================================================================
# Load data
# =============================================================================

file_list_sim = glob.glob(dir_in + "/mirs_img_*.nc")

in_file = file_list_sim[0]
fh = Dataset(in_file, mode='r')
filename = os.path.basename(in_file)
in_file_obs_list = glob.glob(dir_obs + "obs_img_" + filename[9:16] + "*.nc")
if len(in_file_obs_list) > 0:
    in_file_obs = in_file_obs_list[0]
    lat_sim    = fh.variables['Latitude'][:]
    lon_sim    = fh.variables['Longitude'][:]
    sfcTyp_sim = fh.variables['Sfc_type'][:]
    tskin_sim  = fh.variables['TSkin'][:]
    tpw_sim    = fh.variables['TPW'][:]
    clw_sim    = fh.variables['CLW'][:]
    angle_sim  = fh.variables['LZ_angle'][:]
    emis_sim   = fh.variables['Emis'][:]
    tbu        = fh.variables['YM'][:]
    fh.close()

    fh = Dataset(in_file_obs, mode='r')
    lat_obs   = fh.variables['Latitude'][:]
    lon_obs   = fh.variables['Longitude'][:]
    tskin_obs = fh.variables['TSkin'][:]
    fh.close()

nfile = 1
for in_file in file_list_sim[1:]:
    fh = Dataset(in_file, mode='r')
    filename = os.path.basename(in_file)
    in_file_obs_list = glob.glob(dir_obs + "obs_img_" + filename[9:16] + "*.nc")
    if len(in_file_obs_list) > 0:
        in_file_obs = in_file_obs_list[0]

        lat1    = fh.variables['Latitude'][:]
        lon1    = fh.variables['Longitude'][:]
        sfcTyp1 = fh.variables['Sfc_type'][:]
        tskin1  = fh.variables['TSkin'][:]
        tpw1    = fh.variables['TPW'][:]
        clw1    = fh.variables['CLW'][:]
        angle1  = fh.variables['LZ_angle'][:]
        emis1   = fh.variables['Emis'][:]
        tbu1    = fh.variables['YM'][:]
        fh.close()

        lat_sim    = np.vstack((lat_sim,    lat1))
        lon_sim    = np.vstack((lon_sim,    lon1))
        sfcTyp_sim = np.vstack((sfcTyp_sim, sfcTyp1))
        tskin_sim  = np.vstack((tskin_sim,  tskin1))
        tpw_sim    = np.vstack((tpw_sim,    tpw1))
        clw_sim    = np.vstack((clw_sim,    clw1))
        angle_sim  = np.vstack((angle_sim,  angle1))
        emis_sim   = np.vstack((emis_sim,   emis1))
        tbu        = np.vstack((tbu,        tbu1))

        fh = Dataset(in_file_obs, mode='r')
        lat2    = fh.variables['Latitude'][:]
        lon2    = fh.variables['Longitude'][:]
        tskin2  = fh.variables['TSkin'][:]
        fh.close()

        lat_obs   = np.vstack((lat_obs,   lat2))
        lon_obs   = np.vstack((lon_obs,   lon2))
        tskin_obs = np.vstack((tskin_obs, tskin2))

        nfile = nfile + 1
        print(nfile, lat_sim.shape, lat_obs.shape)
        print(in_file)
        print(in_file_obs)

print("No. of input files: " + str(nfile))
print("========Reading in is DONE!========")

lat_sim    = np.asarray(lat_sim)
lon_sim    = np.asarray(lon_sim)
sfcTyp_sim = np.asarray(sfcTyp_sim)
tskin_sim  = np.asarray(tskin_sim)
tpw_sim    = np.asarray(tpw_sim)
clw_sim    = np.asarray(clw_sim)
angle_sim  = np.asarray(angle_sim)
emis_sim   = np.asarray(emis_sim)
tbu        = np.asarray(tbu)

lat_obs    = np.asarray(lat_obs)
lon_obs    = np.asarray(lon_obs)
tskin_obs  = np.asarray(tskin_obs)

if (lat_sim.shape != lat_obs.shape):
    print("STOPPED! Array shape not equal between simulation and observation!")
    print("No. of Files (simulation, observation): ", lat_sim.shape[0], lat_obs.shape[0])
    print("Data array shape (simulation, observation):", lat_sim[0][:][:].shape, lat_obs[0,:,:].shape)
    sys.exit()


# =============================================================================
# Ocean mask
# =============================================================================

mask = [(sfcTyp_sim == 0) & (tskin_sim > 0) & (tskin_obs > 0)]

lat_sim_ocean   = lat_sim[tuple(mask)]
lon_sim_ocean   = lon_sim[tuple(mask)]
tskin_sim_ocean = tskin_sim[tuple(mask)]
tpw_sim_ocean   = tpw_sim[tuple(mask)]
clw_sim_ocean   = clw_sim[tuple(mask)]
angle_sim_ocean = angle_sim[tuple(mask)]
emis_sim_ocean  = emis_sim[tuple(mask)]
tbu_ocean       = tbu[tuple(mask)]

lat_obs_ocean   = lat_obs[tuple(mask)]
lon_obs_ocean   = lon_obs[tuple(mask)]
tskin_obs_ocean = tskin_obs[tuple(mask)]

angle_sim_ocean_cos = np.cos(angle_sim_ocean)


# =============================================================================
# Assemble predictors and target
#
# Two experiments from Liu et al. (2022):
#
#   DNN-TB (active below):
#     Target:     observed SST (tskin_obs)
#     Predictors: lon, lat, brightness temps (22 ch), cos(zenith) — 25 vars
#     Result:     std dev 2.15 K (Jan), 2.27 K (Jul)
#
#   DNN-Retrieval (commented out below):
#     Target:     SST residual = MiRS retrieved SST - observed SST
#     Predictors: lon, lat, MiRS TSkin, emissivity (22 ch), CLW, TPW,
#                 cos(zenith) — 28 vars
#     Result:     std dev 1.80 K (Jan), 1.92 K (Jul)  ← best performing
#
# To run DNN-Retrieval: comment out the DNN-TB block and uncomment DNN-Retrieval.
# =============================================================================

# -----------------------------------------------------------------------------
# DNN-TB: predict SST from brightness temperatures (25 inputs)
# -----------------------------------------------------------------------------
data_org = np.concatenate((
    np.array([tskin_obs_ocean]),        # target: observed SST
    np.array([lon_sim_ocean]),
    np.array([lat_sim_ocean]),
    tbu_ocean.T,                        # 22 brightness temp channels
    np.array([angle_sim_ocean_cos])
), axis=0)
data_org = data_org.T

col_names = ['tskin_obs', 'lon', 'lat',
             'tbu1','tbu2','tbu3','tbu4','tbu5','tbu6','tbu7','tbu8',
             'tbu9','tbu10','tbu11','tbu12','tbu13','tbu14','tbu15',
             'tbu16','tbu17','tbu18','tbu19','tbu20','tbu21','tbu22',
             'angle']

# -----------------------------------------------------------------------------
# DNN-Retrieval: predict SST residual from MiRS physical retrievals (28 inputs)
# Reference: Liu et al. (2022) Table II — best performing experiment
# Uncomment this block and comment out the DNN-TB block above to run.
# -----------------------------------------------------------------------------
# tskin_residual = tskin_sim_ocean - tskin_obs_ocean   # MiRS retrieved - observed
#
# data_org = np.concatenate((
#     np.array([tskin_residual]),         # target: SST residual
#     np.array([lon_sim_ocean]),
#     np.array([lat_sim_ocean]),
#     np.array([tskin_sim_ocean]),        # MiRS retrieved SST
#     emis_sim_ocean.T,                   # 22 emissivity channels
#     np.array([clw_sim_ocean]),
#     np.array([tpw_sim_ocean]),
#     np.array([angle_sim_ocean_cos])
# ), axis=0)
# data_org = data_org.T
#
# col_names = ['tskin_residual', 'lon', 'lat', 'tskin_sim',
#              'emis1','emis2','emis3','emis4','emis5','emis6','emis7','emis8',
#              'emis9','emis10','emis11','emis12','emis13','emis14','emis15',
#              'emis16','emis17','emis18','emis19','emis20','emis21','emis22',
#              'clw', 'tpw', 'angle']
# -----------------------------------------------------------------------------

print(data_org.shape)
rows, cols = np.where(data_org < -900)
data = np.delete(data_org, np.unique(rows), 0)
print(data.shape)

X = data[:, 1:]   # predictors
y = data[:, 0]    # target


# =============================================================================
# Train / test split and normalization
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

train_dataset = pd.DataFrame(X_train, columns=col_names[1:])
test_dataset  = pd.DataFrame(X_test,  columns=col_names[1:])
train_labels  = pd.DataFrame(y_train, columns=[col_names[0]])
test_labels   = pd.DataFrame(y_test,  columns=[col_names[0]])

train_stats = train_dataset.describe().transpose()
print(train_stats)

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data  = norm(test_dataset)


# =============================================================================
# Model
# =============================================================================

def build_model():
    model = keras.Sequential([
        layers.Dense(200, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.title('Learning Curves\nTraining date:' + date)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [Tskin bias]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],     label='Training')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Validation')
    plt.legend()
    plt.savefig(out_dir + 'MAE_model_node200_patience' + str(patience) + '.png')
    plt.clf()

    plt.figure()
    plt.title('Learning Curves\nTraining date:' + date)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Tskin bias^2]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],     label='Training')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Validation')
    plt.legend()
    plt.savefig(out_dir + 'MSE_model_node200_patience' + str(patience) + '.png')
    plt.close()
    plt.clf()


# =============================================================================
# Train
# =============================================================================

EPOCHS  = 1000
patience = 100

tensorboard_cb = keras.callbacks.TensorBoard(
    os.path.join(out_dir, 'keras_tensorboard'),
    histogram_freq=1)

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
    model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, batch_size=100000,
                    validation_split=0.2, verbose=1,
                    callbacks=[early_stop, tensorboard_cb])

# Save model
model.save(entire_model_name)
model_json = model.to_json()
with open(model_json_name, "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_weights_name)
print("Saved model to disk")

hist = pd.DataFrame(history.history)
np.savetxt(history_name, hist, fmt='%10.4f', delimiter="")

plot_history(history)


# =============================================================================
# Evaluate
# =============================================================================

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} K".format(mae))