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

#from tensorflow.keras.utils import plot_model
np.set_printoptions(threshold=sys.maxsize)


date     = "12days"
dir_in   = "/ccsopen/proj/gen175/mirs/data/daily_data/"

file_list_sim = glob.glob(dir_in + "/mirs_img_*.nc")
dir_obs       = dir_in

out_dir       = "/gpfs/wolf/gen175/proj-shared/mirs/sliu//4GPU/"


entire_model_name = out_dir + "Entire_Model_node200_"+ date +".h5"
model_json_name   = out_dir + "model_node200_"+ date +".json"
model_weights_name= out_dir + "model_node200_"+ date +".h5"
history_name      = out_dir + "history_model_node200_"+ date +".csv"
#
#
#

nfov  = 96
nchan = 22

in_file = file_list_sim[0]
fh = Dataset(in_file, mode='r')
filename=os.path.basename(in_file)
in_file_obs_list = glob.glob(dir_obs + "obs_img_" + filename[9:16] + "*.nc")
if len(in_file_obs_list) > 0:
   in_file_obs = in_file_obs_list[0]
   lat_sim     = fh.variables['Latitude'][:]
   lon_sim     = fh.variables['Longitude'][:]
   sfcTyp_sim  = fh.variables['Sfc_type'][:]
   tskin_sim   = fh.variables['TSkin'][:]
   tpw_sim     = fh.variables['TPW'][:]
   clw_sim     = fh.variables['CLW'][:]
   angle_sim   = fh.variables['LZ_angle'][:]
   emis_sim    = fh.variables['Emis'][:]
   tbu         = fh.variables['YM'][:]
   fh.close()

   fh = Dataset(in_file_obs, mode='r')
   lat_obs     = fh.variables['Latitude'][:]
   lon_obs     = fh.variables['Longitude'][:]
   tskin_obs   = fh.variables['TSkin'][:]
   fh.close()

nfile = 1
for in_file in file_list_sim[1:]:
   fh = Dataset(in_file, mode='r')
   filename=os.path.basename(in_file)
   in_file_obs_list = glob.glob(dir_obs + "obs_img_" + filename[9:16] + "*.nc")
   if len(in_file_obs_list) > 0:
      in_file_obs = in_file_obs_list[0]

      lat1     = fh.variables['Latitude'][:]
      lon1     = fh.variables['Longitude'][:]
      sfcTyp1  = fh.variables['Sfc_type'][:]
      tskin1   = fh.variables['TSkin'][:]
      tpw1     = fh.variables['TPW'][:]
      clw1     = fh.variables['CLW'][:]
      angle1   = fh.variables['LZ_angle'][:]
      emis1    = fh.variables['Emis'][:]
      tbu1     = fh.variables['YM'][:]
      fh.close()

      lat_sim     = np.vstack((lat_sim,lat1))
      lon_sim     = np.vstack((lon_sim,lon1))
      sfcTyp_sim  = np.vstack((sfcTyp_sim,sfcTyp1))
      tskin_sim   = np.vstack((tskin_sim,tskin1))
      tpw_sim     = np.vstack((tpw_sim,tpw1))
      clw_sim     = np.vstack((clw_sim,clw1))
      angle_sim   = np.vstack((angle_sim,angle1))
      emis_sim    = np.vstack((emis_sim,emis1))
      tbu         = np.vstack((tbu,tbu1))

      fh = Dataset(in_file_obs, mode='r')
      lat2     = fh.variables['Latitude'][:]
      lon2     = fh.variables['Longitude'][:]
      tskin2   = fh.variables['TSkin'][:]
      fh.close()

      lat_obs   = np.vstack((lat_obs,lat2))
      lon_obs   = np.vstack((lon_obs,lon2))
      tskin_obs = np.vstack((tskin_obs,tskin2))

      nfile = nfile + 1
      print(nfile,lat_sim.shape,lat_obs.shape)
      print(in_file)
      print(in_file_obs)

print("No. of input files: " + str(nfile))
print("========Reading in is DONE!========")

lat_sim     = np.asarray(lat_sim)
lon_sim     = np.asarray(lon_sim)
sfcTyp_sim  = np.asarray(sfcTyp_sim)
tskin_sim   = np.asarray(tskin_sim)
tpw_sim     = np.asarray(tpw_sim)
clw_sim     = np.asarray(clw_sim)
angle_sim   = np.asarray(angle_sim)
emis_sim    = np.asarray(emis_sim)
tbu         = np.asarray(tbu)

lat_obs     = np.asarray(lat_obs)
lon_obs     = np.asarray(lon_obs)
tskin_obs   = np.asarray(tskin_obs)

if (lat_sim.shape != lat_obs.shape):
   print("STOPPED! Array shape not equal between simulation and observation!")
   print("No. of Files (simulation, observation): ", lat_sim.shape[0], lat_obs.shape[0])
   print("Data array shape (simulation, observation):", lat_sim[0][:][:].shape,lat_obs[0,:,:].shape)
   sys.exit()
#
#---keep ocean samples only
#
mask = [(sfcTyp_sim == 0) & (tskin_sim > 0) & (tskin_obs > 0) ]
lat_sim_ocean     = lat_sim[tuple(mask)]
lon_sim_ocean     = lon_sim[tuple(mask)]
tskin_sim_ocean   = tskin_sim[tuple(mask)]
tpw_sim_ocean     = tpw_sim[tuple(mask)]
clw_sim_ocean     = clw_sim[tuple(mask)]
angle_sim_ocean   = angle_sim[tuple(mask)]
emis_sim_ocean    = emis_sim[tuple(mask)]
tbu_ocean         = tbu[tuple(mask)]
#jday_ocean        = jday[tuple(mask)]

lat_obs_ocean     = lat_obs[tuple(mask)]
lon_obs_ocean     = lon_obs[tuple(mask)]
tskin_obs_ocean   = tskin_obs[tuple(mask)]

#print(emis_sim_ocean.shape)

angle_sim_ocean_cos  = np.cos(angle_sim_ocean)
#
#=================================================================
#
data_org  = np.concatenate((np.array([tskin_obs_ocean]),np.array([lon_sim_ocean]),np.array([lat_sim_ocean]),tbu_ocean.T,\
                            np.array([angle_sim_ocean_cos])),axis=0)
data_org  = data_org.T
print(data_org.shape)
rows,cols   = np.where(data_org < -900)
data=np.delete(data_org, np.unique(rows) ,0)
print(data.shape)

X = data[:,1:26] # Predictors
y = data[:,0]   # Target

##### Split to train/test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# training dataset
train_dataset  = pd.DataFrame({'lon':X_train[:,0],'lat':X_train[:,1],'tbu1':X_train[:,2],'tbu2':X_train[:,3], \
                               'tbu3':X_train[:,4],'tbu4':X_train[:,5],'tbu5':X_train[:,6],'tbu6':X_train[:,7], \
                               'tbu7':X_train[:,8],'tbu8':X_train[:,9],'tbu9':X_train[:,10],'tbu10':X_train[:,11],\
                               'tbu11':X_train[:,12],'tbu12':X_train[:,13],'tbu13':X_train[:,14],\
                               'tbu14':X_train[:,15],'tbu15':X_train[:,16],'tbu16':X_train[:,17],\
                               'tbu17':X_train[:,18],'tbu18':X_train[:,19],'tbu19':X_train[:,20],\
                               'tbu20':X_train[:,21],'tbu21':X_train[:,22],'tbu22':X_train[:,23],\
                               'angle':X_train[:,24] })
train_labels   = pd.DataFrame(y_train,columns=['Tskin_obs'])

# test dataset
test_dataset  = pd.DataFrame({'lon':X_test[:,0],'lat':X_test[:,1],'tbu1':X_test[:,2],'tbu2':X_test[:,3], \
                              'tbu3':X_test[:,4],'tbu4':X_test[:,5],'tbu5':X_test[:,6],'tbu6':X_test[:,7], \
                              'tbu7':X_test[:,8],'tbu8':X_test[:,9],'tbu9':X_test[:,10],'tbu10':X_test[:,11],\
                              'tbu11':X_test[:,12],'tbu12':X_test[:,13],'tbu13':X_test[:,14],\
                              'tbu14':X_test[:,15],'tbu15':X_test[:,16],'tbu16':X_test[:,17],\
                              'tbu17':X_test[:,18],'tbu18':X_test[:,19],'tbu19':X_test[:,20],\
                              'tbu20':X_test[:,21],'tbu21':X_test[:,22],'tbu22':X_test[:,23],\
                              'angle':X_test[:,24] } )
test_labels    = pd.DataFrame(y_test,columns=['Tskin_obs'])

# look at the overall statistics
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
print(train_stats)

# Normalize the data
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# look at the overall statistics of normed train dataset
train_stats = normed_train_data.describe()
train_stats = train_stats.transpose()
#print(train_stats)

############### Build the model ##############################################################

def build_model():
  model = keras.Sequential([
    layers.Dense(200, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(200, activation=tf.nn.relu),
#    layers.Dense(22)
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

#Inspect the model
#model = build_model()
#model.summary()

############### Train the model ###############################################################
### Train the model for upto 1000 epochs, and record the training and validation accuracy in the history object.
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#  Visualize results
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.title('Learning Curves\nTraining date:' + date)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Tkin bias]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Training')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Validation')
#  plt.ylim([0,2])
  plt.legend()
  plt.savefig(out_dir + 'MAE_model_node200_patience'+str(patience)+'.png')
  plt.clf()
  
  plt.figure()
  plt.title('Learning Curves\nTraining date:' + date)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [Tskin bias^2]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Training')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Validation')
#  plt.ylim([0,2])
  plt.legend()
# plt.show()
  plt.savefig(out_dir + 'MSE_model_node200_patience'+str(patience)+'.png')
  plt.close()
  plt.clf()

#plot_history(history)

#################
#### This graph shows little improvement, or even degradation in the validation error after about 100 epochs. Let's update the model.fit call to automatically stop training when the validation score doesn't improve. We'll use an EarlyStopping callback that tests a training condition for every epoch. If a set amount of epochs elapses without showing improvement, then automatically stop the training.
#################

EPOCHS = 1000
patience=100 
### train the model again using an EarlyStopping callback

# Set up Tensoboard for logging
tensorboard_cb = keras.callbacks.TensorBoard(
         os.path.join(out_dir, 'keras_tensorboard'),
         histogram_freq=1)


#---GPU test
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
with strategy.scope():
   model = build_model()
#---GPU test done

## The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
#history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,batch_size=100000,
#                    validation_split = 0.2, verbose=1, callbacks=[early_stop, PrintDot()])
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,batch_size=100000,
                     validation_split = 0.2, verbose=1, callbacks=[early_stop, tensorboard_cb, PrintDot()])

"""
filepath        = "checkpoint_best_model_node200_epoch"+str(EPOCHS)+".h5"
checkpoint      = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
history         = model.fit(normed_train_data, train_labels, epochs=EPOCHS, batch_size=1000, validation_split = 0.2, verbose = 1, callbacks = [checkpoint])
# load saved best model
best_model      = load_model(filepath)
print('best_model loaded')
# history         = best_model.fit(normed_train_data, train_labels, epochs=EPOCHS, batch_size=1000, validation_split = 0.2, verbose = 1, callbacks = [checkpoint])
"""


# Save the entire model to a HDF5 file
#model.save("Entire_Model_node200_patience"+str(patience)+".h5") 
model.save(entire_model_name) 

# serialize model to JSON
model_json = model.to_json()
#with open("model_node200_patience"+str(patience)+".json", "w") as json_file:
with open(model_json_name, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model_node200_patience"+str(patience)+".h5")
model.save_weights(model_weights_name)
print("Saved model to disk")

hist = pd.DataFrame(history.history)
hist.tail() # print out final hist results
#np.savetxt('history_model_node200_patience'+str(patience)+'.csv',hist,fmt='%10.4f',delimiter="")
np.savetxt(history_name,hist,fmt='%10.4f',delimiter="")

# plot MSE and MAS 
plot_history(history)

#### Make predictions
## test dataset 
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
sys.exit()

#plot_model(model, 'model.png', show_shapes=True)

test_predictions = model.predict(normed_test_data)
test_predictions = model.predict(normed_test_data).flatten()
#print(test_predictions)
nbins = 41
bins  = np.linspace(-10.,10.,nbins)

x1 = test_predictions
x2 = np.array([y_test]).flatten()

plt.figure()
plt.hist([x1,x2],bins, \
         weights=[100.*np.ones(len(x1))/len(x1),100.*np.ones(len(x2))/len(x2)], \
         histtype='step', linewidth=2, label=["Predict","Truth"])

plt.legend(loc='best')
plt.title('Skin Temperature Bias (over Ocean) histogram\nTraining date:' + date)
plt.xlabel('Tskin Bias (K)')
plt.ylabel('Frequency (%)')
plt.savefig('Validation_hist_step.png')

plt.clf()

plt.figure()
plt.hist([x1,x2],bins, \
         weights=[100.*np.ones(len(x1))/len(x1),100.*np.ones(len(x2))/len(x2)], \
         rwidth=0.85,label=["Predict","Truth"])

plt.legend(loc='best')
plt.title('Skin Temperature Bias (over Ocean) histogram\nTraining date:' + date)
plt.xlabel('Tskin Bias (K)')
plt.ylabel('Frequency (%)')
plt.savefig('Validation_hist_bar.png')

plt.clf()
