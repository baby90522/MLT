#%% ---------- Package ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from Model import saveModel, modelEvaluation, ArrayDataset
from Model import MTLNet, modelMTLTrain, modelMTLValidate


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#%% ---------- Data ----------
# dealing with data types
int_features = ['Year', 'Month', 'Day', 'Hour']
cat_features = ['Season', 'Weekday', 'Weekend']
num_features = ['Temperature', 'Pressure', 'Humidity', 'Wind_Speed', 'Rainfall']
dtype_dict = {c: 'category' for c in cat_features}
dtype_dict.update({c: np.int32 for c in int_features})
dtype_dict.update({c: np.float32 for c in num_features})

# load data
df = pd.read_csv('data_NA.csv', dtype = dtype_dict)
#df.info()

# set Time (Date: yyyy-mm-dd hh:mm:ss) as index
df['Time'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) + ' ' + \
             df['Hour'].astype(str).apply(lambda e: '{:02d}'.format(int(e)))
df['Time'] = pd.to_datetime(df['Time'], format = '%Y-%m-%d %H')
df.set_index('Time', inplace = True)
df = df.sort_index()
#df


#%% ---------- Preprocessing ----------
# split train & test
filt = (df.index >= '2021-01-01')
df_train, df_test = df.loc[~filt], df.loc[filt]

# data imputation
df_train = df_train.fillna(method = 'ffill')
df_test = df_test.fillna(method = 'ffill')

# drop outliners (Load < 1000)
filt = (df_train['Load'] < 1000)
df_train = df_train.drop(df_train.loc[filt].index)

# main label: Next_Hour_Load, aux label: Next_Hour_Temperature
df_train['Next_Hour_Load'] = df_train['Load'].shift(-1)
df_train['Next_Hour_Temperature'] = df_train['Temperature'].shift(-1)
df_train = df_train.fillna(method = 'ffill')

df_test['Next_Hour_Load'] = df_test['Load'].shift(-1)
df_test['Next_Hour_Temperature'] = df_test['Temperature'].shift(-1)
df_test = df_test.fillna(method = 'ffill')

features = ['Year', 'Month', 'Day', 'Hour', 'Season', 'Weekday', 'Weekend',
            'Temperature', 'Pressure', 'Humidity', 'Wind_Speed', 'Rainfall']
main_label = 'Next_Hour_Load'
aux_label = 'Next_Hour_Temperature'

# split X & y
X = df_train[features]
y_main = df_train[main_label]
y_aux = df_train[aux_label]

X_test = df_test[features]
y_test_main = df_test[main_label]
y_test_aux = df_test[aux_label]

# split train & validation
filt = (X.index >= '2020-11-01')
X_train, X_val = X.loc[~filt], X.loc[filt]
y_train_main, y_val_main = y_main.loc[~filt], y_main.loc[filt]
y_train_aux, y_val_aux = y_aux.loc[~filt], y_aux.loc[filt]

# plot
fig, ax = plt.subplots(figsize = (22, 8))
ax.plot(y_train_main, label = 'Training Data')
ax.plot(y_val_main, label = 'Validating Data')
ax.plot(y_test_main, label = 'Testing Data')
plt.legend()

# normalization (num) & one-hot encoding (cat)
num_transformer = Pipeline(steps = [('scaler', MinMaxScaler(feature_range = (-1, 1)))])
cat_transformer = Pipeline(steps = [('ohe', OneHotEncoder(handle_unknown = 'ignore'))])
preprocessor = ColumnTransformer(transformers = [('num', num_transformer, num_features),
                                                 ('cat', cat_transformer, cat_features)],
                                 remainder = 'passthrough')

X = preprocessor.fit_transform(X)
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# torch: Dataset, DataLoader (MTL)
train_val_dataset_MTL = ArrayDataset(X, y_main, y_aux, True)
train_dataset_MTL = ArrayDataset(X_train, y_train_main, y_train_aux, True)
val_dataset_MTL = ArrayDataset(X_val, y_val_main, y_val_aux, True)
test_dataset_MTL = ArrayDataset(X_test, y_test_main, y_test_aux, True)
print(f'training size: {len(train_dataset_MTL)}, validation size: {len(val_dataset_MTL)}, test size:{len(test_dataset_MTL)}')

batch_size = 64
train_val_loader = DataLoader(dataset = train_val_dataset_MTL, batch_size = batch_size, shuffle = True)
train_loader = DataLoader(dataset = train_dataset_MTL, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset_MTL, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset_MTL, batch_size = batch_size, shuffle = False)

'''
# torch: Dataset, DataLoader
train_val_dataset = ArrayDataset(X, y_main, y_aux, False)
train_dataset = ArrayDataset(X_train, y_train_main, y_train_aux, False)
val_dataset = ArrayDataset(X_val, y_val_main, y_val_aux, False)
test_dataset = ArrayDataset(X_test, y_test_main, y_test_aux, False)
print(f'training size: {len(train_dataset)}, validation size: {len(val_dataset)}, test size:{len(test_dataset)}')

train_val_loader = DataLoader(dataset = train_val_dataset, batch_size = batch_size, shuffle = True)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
'''


#%% ---------- Model_MTL ----------
input_size = X_train.shape[1]
print(f'input size {input_size}')

num_of_ensembles = 10
epochs = 500
warm_up = 50
weight = 165
model_file_names = []

for ensemble in range(num_of_ensembles):
    print(f'Ensemble Step {ensemble + 1}')
    train_losses = []
    val_losses = []
    early_stop = False
    model_saving = True

    model = MTLNet(input_size).to(device)
    
    prev_mean_val_losses = 0
    count = 0
    
    training_over = False

    learning_rate = 0.05
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 1e-08, weight_decay = 0.0001, amsgrad = False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.05, patience = 5, verbose = True)
    
    for epoch in np.arange(epochs):
        train_losses, mean_loss, mean_loss_main, mean_loss_aux =  modelMTLTrain(model, train_loader, device, optimizer, scheduler, weight, train_losses, False)
        valid_losses = []

        mean_val_losses = modelMTLValidate(model, val_loader, device, val_losses, valid_losses)        
        if epoch >= warm_up:
            if epoch == warm_up:
                prev_mean_val_losses = 9999999
            elif mean_val_losses > prev_mean_val_losses:
                count += 1
            else:
                prev_mean_val_losses = mean_val_losses
                count = 0
                model_saving = True

            if count >= 30:
                print('================ Early stop ================')
                print(f'mean validation loss: {prev_mean_val_losses:.2f}')
                early_stop = True
                count = 0
                training_over = True
                break        

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: average training loss is {mean_loss:.2f}, validation loss is {mean_val_losses:.2f}, main task loss: {mean_loss_main:.2f}, auxiliary task loss: {mean_loss_aux:.2f}')
    
    if training_over:
        learning_rate = 0.0001
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, betas = (0.9, 0.99), eps = 1e-08, weight_decay = 0.0001, amsgrad = False)        

        for e in np.arange(20):
            train_losses, mean_loss, mean_loss_main, mean_loss_aux =  modelMTLTrain(model, train_loader, device, optimizer, scheduler, weight, train_losses, True)
            print(f'Epoch {e+1}: average training loss is {mean_loss:.2f}, main task loss: {mean_loss_main:.2f}, auxiliary task loss: {mean_loss_aux:.2f}')

        now = datetime.now()
        timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
        model_filePath = f'./model/model_{epoch + 1}_{timestamp}.pth'
        print(f"Saving the model...model name: {model_filePath}")
        state_dict_filePath = f'./model/state_dict_ensemble_{epoch + 1}_{timestamp}.pth'
        saveModel(model, model_filePath, state_dict_filePath)   
        
        model_file_names.append(model_filePath)


#%% ---------- Results_MTL ----------
# training
fig, ax = plt.subplots(figsize = (10, 8))
train_losses = train_losses[1:]

ax.plot(range(1, len(train_losses)+1), train_losses, label = 'Training loss')
ax.plot(range(1, len(val_losses)+1), val_losses, label = 'Validation loss')
plt.legend()

# testing
print()
ensemble_MTL_predictions = []
MTL_MAE_list = []
MTL_MSE_list = []
MTL_RMSE_list = []

for f in model_file_names:
    model = torch.load(f)
    MTL_predictions = modelEvaluation(model, test_loader, device, True)
    MTL_MAE = mean_absolute_error(y_test_main, MTL_predictions)
    MTL_MSE = mean_squared_error(y_test_main, MTL_predictions)
    MTL_RMSE = np.sqrt(MTL_MSE)
    
    MTL_MAE_list.append(MTL_MAE)
    MTL_MSE_list.append(MTL_MSE)
    MTL_RMSE_list.append(MTL_RMSE)
    print(f'MTL model {f}, MAE: {MTL_MAE:.2f}, MSE: {MTL_MSE:.2f}, RMSE: {MTL_RMSE:.2f}')
    
    ensemble_MTL_predictions.append(MTL_predictions)
    
print(f'Average Performance for {len(MTL_MAE_list)} MTL models is: MAE = {np.mean(MTL_MAE_list):.2f}, MSE = {np.mean(MTL_MSE_list):.2f}, RMSE = {np.mean(MTL_RMSE_list):.2f}')
print(f'Performance Standard Deviation for {len(MTL_MAE_list)} MTL models is: MAE = {np.std(MTL_MAE_list):.2f}, MSE = {np.std(MTL_MSE_list):.2f}, RMSE = {np.std(MTL_RMSE_list):.2f}')

final_MTL_predictions = [np.mean(MTL_predictions) for MTL_predictions in zip(*ensemble_MTL_predictions)]
MTL_MAE = mean_absolute_error(y_test_main, final_MTL_predictions)
MTL_MSE = mean_squared_error(y_test_main, final_MTL_predictions)
MTL_RMSE = np.sqrt(MTL_MSE)
print(f'Ensemble {len(ensemble_MTL_predictions)} MTL Model performance: MAE = {MTL_MAE:.2f}, MSE = {MTL_MSE:.2f}, RMSE = {MTL_RMSE:.2f}')

final_MTL_predictions = pd.DataFrame(final_MTL_predictions)
final_MTL_predictions = final_MTL_predictions.set_index(y_test_main.index.get_level_values('Time'))
fig, ax = plt.subplots(figsize = (22, 8))
ax.plot(y_test_main, label = 'Real Load')
ax.plot(final_MTL_predictions, label = 'Predicted Load')
plt.legend()
