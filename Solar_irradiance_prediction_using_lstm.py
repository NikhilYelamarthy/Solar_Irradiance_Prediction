{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: dcor in d:\\new folder\\lib\\site-packages (0.4)\n",
      "Requirement already satisfied: setuptools in d:\\new folder\\lib\\site-packages (from dcor) (45.2.0.post20200210)\n",
      "Requirement already satisfied: numpy in d:\\new folder\\lib\\site-packages (from dcor) (1.18.1)\n",
      "Requirement already satisfied: numba in d:\\new folder\\lib\\site-packages (from dcor) (0.48.0)\n",
      "Requirement already satisfied: scipy in d:\\new folder\\lib\\site-packages (from dcor) (1.4.1)\n",
      "Requirement already satisfied: llvmlite<0.32.0,>=0.31.0dev0 in d:\\new folder\\lib\\site-packages (from numba->dcor) (0.31.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install dcor\n",
    "from pylab import *\n",
    "import numpy as np \n",
    "import pandas as pd \n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from datetime import datetime\n",
    "import math\n",
    "from IPython.display import display\n",
    "import scipy.optimize as opt\n",
    "import statistics\n",
    "import os\n",
    "from scipy.spatial.distance import  pdist, squareform\n",
    "import dcor\n",
    "from statsmodels.graphics import tsaplots\n",
    "from statsmodels.tsa import stattools \n",
    "import scipy.ndimage.interpolation as sp\n",
    "import sklearn.model_selection as skl\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from keras.models import Sequential,save_model,load_model\n",
    "from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding,RepeatVector,BatchNormalization\n",
    "from keras.utils import *\n",
    "from keras.callbacks import ModelCheckpoint\n",
    "from keras.losses import *\n",
    "from keras.optimizers import *\n",
    "import os\n",
    "\n",
    "\n",
    "os.chdir(\"D:\\Solar_prediction_using_lstm\")\n",
    "BATCH_SIZE = 17"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess(df):\n",
    "  df['hour'] = pd.to_datetime(df['period_end_IST'],format=\"%Y-%m-%dT%H:%M:%S\").dt.hour\n",
    "  df['month'] = pd.to_datetime(df['period_end_IST'],format=\"%Y-%m-%dT%H:%M:%S\").dt.month\n",
    "  df['year'] = pd.to_datetime(df['period_end_IST'],format=\"%Y-%m-%dT%H:%M:%S\").dt.year\n",
    "  df['hourcosine']=np.cos(df['hour']*(np.pi)/12)\n",
    "  df['hoursine']=np.sin(df['hour']*(np.pi)/12)\n",
    "  df['monthcosine']=np.cos(df['month']*(np.pi)/6)\n",
    "  df['monthsine']=np.sin(df['month']*(np.pi)/6)\n",
    " \n",
    "  #print(df)\n",
    "  df.to_csv('test.csv',index=False,header=True)\n",
    "  \n",
    "  A= df.to_numpy(dtype=None, copy=False)   \n",
    "  #print(A)\n",
    "\n",
    "  X=A[:,[ 2,3,5,6,7,8,9,10,11,12,13,14,15,16]]\n",
    "  Y=A[:,4]\n",
    "  #print(X[:,0])\n",
    "  #print(X,Y)\n",
    "\n",
    "  df.drop(df.columns[[0,1,4]], axis=1, inplace=True)\n",
    "  \n",
    "  featureindex= { i:feature for(i,feature) in enumerate(df.columns)}\n",
    "  #print(featureindex)\n",
    "  \n",
    "  \n",
    "  #for i in range(len(featureindex)):\n",
    "  # print (featureindex[i],'\\n',dcor.distance_correlation(X[:,i].astype(float),Y.astype(float)),np.corrcoef(X[:,i].astype(float),Y.astype(float)))\n",
    " \n",
    " \n",
    "\n",
    "  return(X,Y)\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def input_output(x,y):\n",
    "  #print('x, y are \\n ',x,y)\n",
    "  xr =x[:,[0,2,3,4,5,6,9,10,11,12,13]]\n",
    "  d=int(xr.shape[0]/24)\n",
    "  m=24\n",
    "  n=xr.shape[1]\n",
    "  #print(d,m,n)\n",
    "\n",
    "  X = np.zeros((d,m,n))\n",
    "  Y = np.zeros((d,m))\n",
    "  for i in range(d):\n",
    "    for j in range(m):\n",
    "      X[i,j,:]=xr[i*m+j,:]\n",
    "      Y[i,j] = y[i*m+j]\n",
    "  \n",
    "  np.savetxt('X.csv', X[:,:,0], delimiter=',', fmt='%f')\n",
    "  np.savetxt('Y.csv', Y, delimiter=',', fmt='%f')\n",
    "  \n",
    "  #X[:,:,2]=X[:,:,2]**3\n",
    "\n",
    "  return(X,Y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def pacf(y):\n",
    "  reversedy=np.flip(y,axis=0)\n",
    "  pacfresults =np.zeros((21,24))\n",
    "  #print(reversedy)\n",
    "  for i in range(6,19):\n",
    "\n",
    "   pacfresults[:,i] = stattools.pacf(reversedy[:,i], nlags=20, method='ywunbiased', alpha=0.05)[0]\n",
    "  np.savetxt('pacfresults.csv', pacfresults, delimiter=',', fmt='%f')\n",
    "  \n",
    "  #fig = tsaplots.plot_pacf(reversedy[:,6].astype(float), lags=100,alpha=0.05)\n",
    "  #plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def addtoX(x,y):\n",
    "   d,m,n =x.shape\n",
    "   temp =np.zeros((d,m,n+1))\n",
    "   xadded=np.zeros((d,m,n+2))\n",
    "   for s in range(2):\n",
    "     yshifted =sp.shift(y,(s+1,0),cval=0)\n",
    "     for i in range(d):\n",
    "      for j in range(m):\n",
    "        if(s==0):\n",
    "         temp[i,j]=np.append(x[i,j],yshifted[i,j])\n",
    "        if(s==1):\n",
    "         xadded[i,j]=np.append(temp[i,j],yshifted[i,j])\n",
    "   #print('x[0]',x[10]) \n",
    "   xadded=np.delete(xadded,(0,1),axis=0) \n",
    "   y=np.delete(y,(0,1),axis=0)  \n",
    "   np.savetxt('Xmodified.csv', xadded[0], delimiter=',', fmt='%f')\n",
    "   \n",
    "   return(xadded,y)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def datasplit(X,Y):\n",
    "   all_indices =list(range(X.shape[0]))\n",
    "   train_indices,test_indices=skl.train_test_split(all_indices,test_size =0.3,shuffle='true')\n",
    "   val_indices, test_indices =skl.train_test_split(test_indices,test_size=0.25,shuffle='true')\n",
    "   X_train  = X [train_indices,:,:]\n",
    "   X_val =X[val_indices,:,:]\n",
    "   X_test=X[test_indices,:,:]\n",
    "   Y_train  = Y[train_indices,:]\n",
    "   Y_val =Y[val_indices,:]\n",
    "   Y_test=Y[test_indices,:]\n",
    "\n",
    "   return(X_train,X_val,X_test,Y_train,Y_val,Y_test)\n",
    "   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-4-a2071cbe6be0>, line 10)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-4-a2071cbe6be0>\"\u001b[1;36m, line \u001b[1;32m10\u001b[0m\n\u001b[1;33m    model.add(LSTM(12,,stateful=False))\u001b[0m\n\u001b[1;37m                      ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "def build_model():\n",
    "  model=Sequential()\n",
    "\n",
    "  model.add(LSTM(48,return_sequences =True,stateful=False,batch_input_shape =(BATCH_SIZE,13,13)))\n",
    "  model.add(Dropout(0.2))\n",
    "\n",
    "  model.add(LSTM(24,return_sequences =False,stateful=True))\n",
    "  model.add(Dropout(0.2))\n",
    "\n",
    "  model.add(LSTM(12,,stateful=False))\n",
    "  model.add(Dropout(0.2))\n",
    "\n",
    "  model.add(BatchNormalization())\n",
    "  model.add(Dense(5))\n",
    "  model.add(RepeatVector(13))\n",
    "  model.add(LSTM(12,return_sequences=True,stateful=False))\n",
    "  model.add(LSTM(24,return_sequences=True))\n",
    "  model.add(LSTM(48,return_sequence=True))\n",
    "  model.add(TimeDistributed(Dense(1)))\n",
    "\n",
    "  return(model)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv('shortdata.csv')\n",
    "X,Y= preprocess(data)\n",
    "X,Y = input_output(X,Y)\n",
    "X,Y = addtoX(X,Y)\n",
    "X=X[:,6:19,:]\n",
    "Y=Y[:,6:19]\n",
    "Xtrain,Xval,Xtest,Ytrain,Yval,Ytest =datasplit(X,Y)\n",
    "#print('xtrain',Xtrain,'ytrain',Ytrain)\n",
    "#print(X,Y)\n",
    "#pacf(Y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3417, 13, 13) (1098, 13, 13) (367, 13, 13) (3417, 13) (1098, 13) (367, 13)\n"
     ]
    }
   ],
   "source": [
    "print(Xtrain.shape,Xval.shape,Xtest.shape,Ytrain.shape,Yval.shape,Ytest.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "np.savetxt('Xtrain.csv', Xtrain[:,:,0], delimiter=',', fmt='%f')\n",
    "np.savetxt('ytrain.csv', Ytrain, delimiter=',', fmt='%f')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "  scaler1 = MinMaxScaler(feature_range=(-1,1))\n",
    "  scaler2 = MinMaxScaler(feature_range=(-1,1))\n",
    "  \n",
    "  Xtrain=scaler1.fit_transform(Xtrain.reshape(-1, Xtrain.shape[-1])).reshape(Xtrain.shape)\n",
    "  Ytrain=scaler2.fit_transform(Ytrain)\n",
    "  Xval=scaler1.transform(Xval.reshape(-1, Xval.shape[-1])).reshape(Xval.shape)\n",
    "  Yval=scaler2.transform(Yval)\n",
    "  #print(Xtrain)\n",
    "  #np.savetxt('./drive/My Drive/Solardata/xtrainnorm.csv', Xtrain[0], delimiter=',', fmt='%f')\n",
    "\n",
    "  Ytrain =Ytrain.reshape(3417,13,1)\n",
    "  Yval =Yval.reshape(1098,13,1)\n",
    " \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    " Yval =Yval[0:1088]\n",
    " Xval =Xval[0:1088]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_3\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "lstm_7 (LSTM)                (17, 24, 48)              11904     \n",
      "_________________________________________________________________\n",
      "dropout_7 (Dropout)          (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "lstm_8 (LSTM)                (17, 24, 48)              18624     \n",
      "_________________________________________________________________\n",
      "dropout_8 (Dropout)          (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "lstm_9 (LSTM)                (17, 24, 48)              18624     \n",
      "_________________________________________________________________\n",
      "dropout_9 (Dropout)          (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "time_distributed_3 (TimeDist (17, 24, 1)               49        \n",
      "=================================================================\n",
      "Total params: 49,201\n",
      "Trainable params: 49,201\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "modelf=build_model()\n",
    "modelf.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 3417 samples, validate on 1088 samples\n",
      "Epoch 1/100\n",
      "3417/3417 [==============================] - 7s 2ms/step - loss: 0.3744 - mean_squared_error: 0.2355 - val_loss: 0.3165 - val_mean_squared_error: 0.1858\n",
      "Epoch 2/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.3120 - mean_squared_error: 0.1776 - val_loss: 0.2880 - val_mean_squared_error: 0.1634\n",
      "Epoch 3/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2944 - mean_squared_error: 0.1619 - val_loss: 0.2770 - val_mean_squared_error: 0.1506\n",
      "Epoch 4/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2866 - mean_squared_error: 0.1546 - val_loss: 0.2712 - val_mean_squared_error: 0.1473\n",
      "Epoch 5/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2792 - mean_squared_error: 0.1511 - val_loss: 0.2706 - val_mean_squared_error: 0.1467mean_squared_err\n",
      "Epoch 6/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2765 - mean_squared_error: 0.1489 - val_loss: 0.2628 - val_mean_squared_error: 0.1419\n",
      "Epoch 7/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2729 - mean_squared_error: 0.1474 - val_loss: 0.2609 - val_mean_squared_error: 0.1400\n",
      "Epoch 8/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2703 - mean_squared_error: 0.1450 - val_loss: 0.2618 - val_mean_squared_error: 0.1406\n",
      "Epoch 9/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2672 - mean_squared_error: 0.1432 - val_loss: 0.2594 - val_mean_squared_error: 0.1402\n",
      "Epoch 10/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2654 - mean_squared_error: 0.1430 - val_loss: 0.2575 - val_mean_squared_error: 0.1411\n",
      "Epoch 11/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2621 - mean_squared_error: 0.1402 - val_loss: 0.2572 - val_mean_squared_error: 0.1413\n",
      "Epoch 12/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2632 - mean_squared_error: 0.1407 - val_loss: 0.2576 - val_mean_squared_error: 0.1397\n",
      "Epoch 13/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2613 - mean_squared_error: 0.1392 - val_loss: 0.2620 - val_mean_squared_error: 0.1498\n",
      "Epoch 14/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2603 - mean_squared_error: 0.1388 - val_loss: 0.2553 - val_mean_squared_error: 0.1383\n",
      "Epoch 15/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2600 - mean_squared_error: 0.1389 - val_loss: 0.2561 - val_mean_squared_error: 0.1371\n",
      "Epoch 16/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2583 - mean_squared_error: 0.1376 - val_loss: 0.2593 - val_mean_squared_error: 0.1451\n",
      "Epoch 17/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2586 - mean_squared_error: 0.1373 - val_loss: 0.2536 - val_mean_squared_error: 0.1376\n",
      "Epoch 18/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2572 - mean_squared_error: 0.1371 - val_loss: 0.2561 - val_mean_squared_error: 0.1402\n",
      "Epoch 19/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2560 - mean_squared_error: 0.1361 - val_loss: 0.2548 - val_mean_squared_error: 0.1378\n",
      "Epoch 20/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2555 - mean_squared_error: 0.1359 - val_loss: 0.2538 - val_mean_squared_error: 0.1379\n",
      "Epoch 21/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2552 - mean_squared_error: 0.1359 - val_loss: 0.2534 - val_mean_squared_error: 0.1398\n",
      "Epoch 22/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2545 - mean_squared_error: 0.1353 - val_loss: 0.2538 - val_mean_squared_error: 0.1405\n",
      "Epoch 23/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2533 - mean_squared_error: 0.1348 - val_loss: 0.2551 - val_mean_squared_error: 0.1368\n",
      "Epoch 24/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2525 - mean_squared_error: 0.1337 - val_loss: 0.2559 - val_mean_squared_error: 0.1376\n",
      "Epoch 25/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2516 - mean_squared_error: 0.1340 - val_loss: 0.2526 - val_mean_squared_error: 0.1388\n",
      "Epoch 26/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2518 - mean_squared_error: 0.1338 - val_loss: 0.2520 - val_mean_squared_error: 0.1358\n",
      "Epoch 27/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2510 - mean_squared_error: 0.1337 - val_loss: 0.2517 - val_mean_squared_error: 0.1349\n",
      "Epoch 28/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2501 - mean_squared_error: 0.1333 - val_loss: 0.2537 - val_mean_squared_error: 0.1375\n",
      "Epoch 29/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2495 - mean_squared_error: 0.1323 - val_loss: 0.2510 - val_mean_squared_error: 0.1395\n",
      "Epoch 30/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2491 - mean_squared_error: 0.1320 - val_loss: 0.2518 - val_mean_squared_error: 0.1372\n",
      "Epoch 31/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2483 - mean_squared_error: 0.1315 - val_loss: 0.2516 - val_mean_squared_error: 0.1383\n",
      "Epoch 32/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2488 - mean_squared_error: 0.1327 - val_loss: 0.2517 - val_mean_squared_error: 0.1397\n",
      "Epoch 33/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2474 - mean_squared_error: 0.1311 - val_loss: 0.2521 - val_mean_squared_error: 0.1389\n",
      "Epoch 34/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2471 - mean_squared_error: 0.1317 - val_loss: 0.2531 - val_mean_squared_error: 0.1370\n",
      "Epoch 35/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2469 - mean_squared_error: 0.1309 - val_loss: 0.2506 - val_mean_squared_error: 0.1355\n",
      "Epoch 36/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2456 - mean_squared_error: 0.1304 - val_loss: 0.2505 - val_mean_squared_error: 0.1403\n",
      "Epoch 37/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2455 - mean_squared_error: 0.1302 - val_loss: 0.2518 - val_mean_squared_error: 0.1427\n",
      "Epoch 38/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2464 - mean_squared_error: 0.1314 - val_loss: 0.2521 - val_mean_squared_error: 0.1404\n",
      "Epoch 39/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2446 - mean_squared_error: 0.1295 - val_loss: 0.2514 - val_mean_squared_error: 0.1402\n",
      "Epoch 40/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2447 - mean_squared_error: 0.1294 - val_loss: 0.2489 - val_mean_squared_error: 0.1391\n",
      "Epoch 41/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2439 - mean_squared_error: 0.1295 - val_loss: 0.2494 - val_mean_squared_error: 0.1389\n",
      "Epoch 42/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2431 - mean_squared_error: 0.1290 - val_loss: 0.2515 - val_mean_squared_error: 0.1390\n",
      "Epoch 43/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2433 - mean_squared_error: 0.1290 - val_loss: 0.2496 - val_mean_squared_error: 0.1383\n",
      "Epoch 44/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2420 - mean_squared_error: 0.1275 - val_loss: 0.2473 - val_mean_squared_error: 0.1372\n",
      "Epoch 45/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2425 - mean_squared_error: 0.1285 - val_loss: 0.2522 - val_mean_squared_error: 0.1408\n",
      "Epoch 46/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2423 - mean_squared_error: 0.1286 - val_loss: 0.2498 - val_mean_squared_error: 0.1368\n",
      "Epoch 47/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2414 - mean_squared_error: 0.1275 - val_loss: 0.2503 - val_mean_squared_error: 0.1381\n",
      "Epoch 48/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2414 - mean_squared_error: 0.1280 - val_loss: 0.2486 - val_mean_squared_error: 0.1378\n",
      "Epoch 49/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2413 - mean_squared_error: 0.1279 - val_loss: 0.2476 - val_mean_squared_error: 0.1362\n",
      "Epoch 50/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2401 - mean_squared_error: 0.1267 - val_loss: 0.2496 - val_mean_squared_error: 0.1380\n",
      "Epoch 51/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2399 - mean_squared_error: 0.1270 - val_loss: 0.2516 - val_mean_squared_error: 0.1395\n",
      "Epoch 52/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2399 - mean_squared_error: 0.1268 - val_loss: 0.2514 - val_mean_squared_error: 0.1428\n",
      "Epoch 53/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2407 - mean_squared_error: 0.1273 - val_loss: 0.2507 - val_mean_squared_error: 0.1385\n",
      "Epoch 54/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2393 - mean_squared_error: 0.1263 - val_loss: 0.2496 - val_mean_squared_error: 0.1386\n",
      "Epoch 55/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2393 - mean_squared_error: 0.1260 - val_loss: 0.2499 - val_mean_squared_error: 0.1359\n",
      "Epoch 56/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2391 - mean_squared_error: 0.1268 - val_loss: 0.2492 - val_mean_squared_error: 0.1380\n",
      "Epoch 57/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2381 - mean_squared_error: 0.1253 - val_loss: 0.2500 - val_mean_squared_error: 0.1389\n",
      "Epoch 58/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2374 - mean_squared_error: 0.1244 - val_loss: 0.2487 - val_mean_squared_error: 0.1402\n",
      "Epoch 59/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2366 - mean_squared_error: 0.1241 - val_loss: 0.2494 - val_mean_squared_error: 0.1411\n",
      "Epoch 60/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2365 - mean_squared_error: 0.1242 - val_loss: 0.2506 - val_mean_squared_error: 0.1391\n",
      "Epoch 61/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2365 - mean_squared_error: 0.1239 - val_loss: 0.2514 - val_mean_squared_error: 0.1415\n",
      "Epoch 62/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2360 - mean_squared_error: 0.1246 - val_loss: 0.2487 - val_mean_squared_error: 0.1394\n",
      "Epoch 63/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2350 - mean_squared_error: 0.1232 - val_loss: 0.2497 - val_mean_squared_error: 0.1408\n",
      "Epoch 64/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2356 - mean_squared_error: 0.1235 - val_loss: 0.2511 - val_mean_squared_error: 0.1426\n",
      "Epoch 65/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2348 - mean_squared_error: 0.1234 - val_loss: 0.2498 - val_mean_squared_error: 0.1408\n",
      "Epoch 66/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2351 - mean_squared_error: 0.1231 - val_loss: 0.2484 - val_mean_squared_error: 0.1389\n",
      "Epoch 67/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2346 - mean_squared_error: 0.1229 - val_loss: 0.2496 - val_mean_squared_error: 0.1415\n",
      "Epoch 68/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2346 - mean_squared_error: 0.1236 - val_loss: 0.2480 - val_mean_squared_error: 0.1396\n",
      "Epoch 69/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2341 - mean_squared_error: 0.1226 - val_loss: 0.2490 - val_mean_squared_error: 0.1393\n",
      "Epoch 70/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2329 - mean_squared_error: 0.1215 - val_loss: 0.2506 - val_mean_squared_error: 0.1472\n",
      "Epoch 71/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2335 - mean_squared_error: 0.1226 - val_loss: 0.2500 - val_mean_squared_error: 0.1376\n",
      "Epoch 72/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2326 - mean_squared_error: 0.1218 - val_loss: 0.2520 - val_mean_squared_error: 0.1385\n",
      "Epoch 73/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2322 - mean_squared_error: 0.1209 - val_loss: 0.2486 - val_mean_squared_error: 0.1412\n",
      "Epoch 74/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2319 - mean_squared_error: 0.1214 - val_loss: 0.2472 - val_mean_squared_error: 0.1392\n",
      "Epoch 75/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2316 - mean_squared_error: 0.1204 - val_loss: 0.2485 - val_mean_squared_error: 0.1407\n",
      "Epoch 76/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2312 - mean_squared_error: 0.1208 - val_loss: 0.2485 - val_mean_squared_error: 0.1386\n",
      "Epoch 77/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2307 - mean_squared_error: 0.1202 - val_loss: 0.2492 - val_mean_squared_error: 0.1413\n",
      "Epoch 78/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2312 - mean_squared_error: 0.1207 - val_loss: 0.2490 - val_mean_squared_error: 0.1394\n",
      "Epoch 79/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2305 - mean_squared_error: 0.1201 - val_loss: 0.2477 - val_mean_squared_error: 0.1387\n",
      "Epoch 80/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2304 - mean_squared_error: 0.1204 - val_loss: 0.2501 - val_mean_squared_error: 0.1442\n",
      "Epoch 81/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2294 - mean_squared_error: 0.1189 - val_loss: 0.2491 - val_mean_squared_error: 0.1445\n",
      "Epoch 82/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2302 - mean_squared_error: 0.1200 - val_loss: 0.2497 - val_mean_squared_error: 0.1390\n",
      "Epoch 83/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2283 - mean_squared_error: 0.1180 - val_loss: 0.2489 - val_mean_squared_error: 0.1386\n",
      "Epoch 84/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2292 - mean_squared_error: 0.1187 - val_loss: 0.2500 - val_mean_squared_error: 0.1441\n",
      "Epoch 85/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2288 - mean_squared_error: 0.1187 - val_loss: 0.2525 - val_mean_squared_error: 0.1455\n",
      "Epoch 86/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2279 - mean_squared_error: 0.1181 - val_loss: 0.2498 - val_mean_squared_error: 0.1419\n",
      "Epoch 87/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2280 - mean_squared_error: 0.1182 - val_loss: 0.2500 - val_mean_squared_error: 0.1436\n",
      "Epoch 88/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2274 - mean_squared_error: 0.1176 - val_loss: 0.2498 - val_mean_squared_error: 0.1416\n",
      "Epoch 89/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2272 - mean_squared_error: 0.1172 - val_loss: 0.2499 - val_mean_squared_error: 0.1427\n",
      "Epoch 90/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2262 - mean_squared_error: 0.1173 - val_loss: 0.2507 - val_mean_squared_error: 0.1407\n",
      "Epoch 91/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2263 - mean_squared_error: 0.1168 - val_loss: 0.2506 - val_mean_squared_error: 0.1433\n",
      "Epoch 92/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2265 - mean_squared_error: 0.1168 - val_loss: 0.2505 - val_mean_squared_error: 0.1426\n",
      "Epoch 93/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2266 - mean_squared_error: 0.1171 - val_loss: 0.2522 - val_mean_squared_error: 0.1465\n",
      "Epoch 94/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2262 - mean_squared_error: 0.1164 - val_loss: 0.2517 - val_mean_squared_error: 0.1441\n",
      "Epoch 95/100\n",
      "3417/3417 [==============================] - 5s 1ms/step - loss: 0.2257 - mean_squared_error: 0.1167 - val_loss: 0.2490 - val_mean_squared_error: 0.1395\n",
      "Epoch 96/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2254 - mean_squared_error: 0.1162 - val_loss: 0.2514 - val_mean_squared_error: 0.1439\n",
      "Epoch 97/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2241 - mean_squared_error: 0.1146 - val_loss: 0.2512 - val_mean_squared_error: 0.1468\n",
      "Epoch 98/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2240 - mean_squared_error: 0.1149 - val_loss: 0.2489 - val_mean_squared_error: 0.1449\n",
      "Epoch 99/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2244 - mean_squared_error: 0.1152 - val_loss: 0.2518 - val_mean_squared_error: 0.1442\n",
      "Epoch 100/100\n",
      "3417/3417 [==============================] - 4s 1ms/step - loss: 0.2240 - mean_squared_error: 0.1144 - val_loss: 0.2537 - val_mean_squared_error: 0.1473\n"
     ]
    },
    {
     "data": {
      "image/png": "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KVN49NFH+f7774mJiSE/P5+UlBRuueUWEhMTueuuuwD47LPPqr1XRUUF3377LTNnzuTBBx/k008/5a9//StNmzZlyZIlLFu2jIyMjBp5LC0t5aabbmLWrFl07dqVsWPHVh3r2bMnc+fOJTIykk8//ZRf//rXvPXWWzz00ENkZWXxzDPPAFBYWFhrOoDMzEweffRRfvnLX4b0OzDGHJgTJkDUR6CWMd6H1xdffMEVV1yBx+OhVatWjBw5knnz5jFkyBCuv/56vF4vF110ERkZGXTu3Jn169dz++23c9555/GDH/ygxvVycnLo1atXre+1atUqli1bxtlnnw2Az+ejTZs2APTv35+rrrqKiy66iIsuuiikvP/oRz8CYPDgwWzYsKHqfu64w3VC69u3L/37969x3nfffUenTp3o1q0bAFdffTUvvPACAAUFBYwfP541a9YgIni93lrfu750LVu2POhqNWNMw06YAFHfk/7KnEISYyJp1yw+bO9f19rfI0aMYO7cubz//vtcc8013H333Vx77bUsXryYjz76iMmTJzN9+nSmTJlS7by4uLg6++urKn369OHrr7+ucez9999n7ty5vPPOOzz88MMsX768wbzHxMQA4PF4qKioqPd+9ldXt9Hf/OY3nH766cyYMYMNGzYwatSoA05XWlpKXFxcSPkwxhy4E74NAlxDdbjbqEeMGMG0adPw+Xzk5uYyd+5chg4dysaNG2nZsiU33XQTN9xwAwsWLCAvLw+/388ll1zCww8/zIIFC2pcr1evXqxdu7ZqOykpiT179gDQo0cPcnNzqwKE1+tl+fLl+P1+Nm/ezOmnn86kSZPIz8+nqKio2rmhOvXUU5k+fToAK1asYOnSpTXS9OzZk++//55169YB8K9//avqWEFBAWlpbu7Gl19+udb7qC8dwOrVq+nbt+8B5dsYEzoLELiG6nD3Yrr44ovp378/AwYM4IwzzmDSpEm0bt2aOXPmkJGRwcCBA3nrrbe444472LJlC6NGjSIjI4MJEybwhz/UXCvp3HPPZe7cuVXbEyZM4JZbbiEjIwOfz8ebb77JPffcw4ABA8jIyOCrr77C5/Nx9dVX069fPwYOHMidd95JSkoKY8aMYcaMGVWN1KG49dZbyc3NpX///jz22GP079+f5OTkamliY2N54YUXOO+88zj11FPp0KFD1bFf/vKX/OpXv+KUU07B59vXxfj0009nxYoVVY3UdaUDmD17Nuedd15I+TXGHDgJtargaJeZman796JZuXJlnfX0wdbuKCJCoHNq+MYshMPFF1/MpEmTqur4jySfz4fX6yU2NpZ169Zx5plnsnr1aqKjo4/I+5eVlTFy5Ei++OILIiMPb01pqH83xhwPRGS+qmbWduyEaYOoT4QQ9iqmcHj00UfJyclplABRXFzM6aefjtfrRVV59tlnj1hwANi0aROPPvroYQ8Oxph97L8L1wbh9fsbOxsHrEePHvTo0aNR3jspKalRl3jt1q1bowRGY04k1gZBZRtEY+fCGGOOLhYgqOzFZBHCGGOCWYDAtUFYCcIYY6qzAIEbzHWsTdZnjDHhZgECV8XkV7VqJmOMCRLWACEio0VklYisFZF7azl+i4gsFZFFIvKFiPQOOtZfRL4OLEm6VERiw5XPiMBsEOEID0dqPYgDNWHCBN58800AbrzxRlasWFEjzcsvv8zEiRPrvc6RWksiOL91CXXK87vuuotZs2YdrqwZc9wKWzdXEfHg1pY+G8gG5onIO6oa/En0uqo+F0h/AfAUMFpEIoF/Ateo6mIRaQ7UPpvb4ckrAH6/EuE5NpacrFwPorZpOA7Uiy++eNDnzpkzh8TERIYPHw64tSQay8svv0zfvn1p27Ztveluv/12brrpJs4444wjlDNjjk3hHAcxFFirqusBRGQqcCFQFSBUNXj+7QT2PcT/AFiiqosD6XYecm4+uBe21ZwvCCDF5yeuwk9EjIfA3K6had0Pzn00pKSqyi9/+Us++OADRIT777+fsWPHkpOTw9ixYyksLKSiooJnn32W4cOHc8MNN5CVlYWIcP3113PnnXdWu17wehArV65k/PjxfPvttwBs2LCBCy64gCVLlvDQQw/x7rvvUlJSwvDhw3n++edrTKA3atSoqqnCX3rpJf7whz/Qpk0bunfvXjVR37vvvssjjzxCeXk5zZs357XXXqOkpITnnnsOj8fDP//5T/7yl7/w2WefVU0dvmjRIm655RaKi4vp0qULU6ZMoWnTpowaNYphw4Yxe/Zs8vPz+fvf/85pp51W4+d1++23M2vWLDp16lSt+q+2e3rrrbeq1sSIi4vj66+/5vHHH6/13jt06MDOnTvZtm0brVu3Dv33bcwJJpxVTGnA5qDt7MC+akTkNhFZB0wCfhrY3R1QEflIRBaISK0T/ovIzSKSJSJZubm5B5/Tys/LMDZBBK8H8emnn3L33XeTk5NTtR5E5bGMjIxq60EsXbqU6667rsb1gteD6NWrF+Xl5axfvx6AadOmcfnllwMwceJE5s2bx7JlyygpKeG9996rM485OTn87ne/48svv+STTz6pVu106qmn8s0337Bw4ULGjRvHpEmT6NixI7fccgt33nknixYtqvEhf+211/LYY4+xZMkS+vXrx4MPPlh1rHKNiaeffrra/kozZsxg1apVLF26lL/97W/VqrFqu6dLL72UzMxMXnvtNRYtWkRcXFy99z5o0CC+/PLLen9nxpzowlmCqO1RvMZHsKpOBiaLyJXA/cD4QL5OBYYAxcBngflCPtvv3BeAF8DNxVRvbup50i8uLmfjrmK6tUwiLtpT72UOVrjXg7j88suZPn069957L9OmTatasW727NlMmjSJ4uJidu3aRZ8+fRgzZkytefzf//7HqFGjSE1NBWDs2LGsXr0agOzs7KoST3l5OZ06dar3fgsKCsjPz2fkyJEAjB8/nssuu6zqeG1rTASbO3du1c+rbdu21aqDQr2n+tLZWhLGNCycJYhsoF3QdjpQ33/kVKByBZts4L+qmqeqxcBMYFBYcsm+Nohw9mJqaD2ItLQ0rrnmGl599VWaNm3K4sWLGTVqFJMnT+bGG2+scd7+60GMHTuW6dOns3r1akSEbt26UVpayq233sqbb77J0qVLuemmm+pcQ6JSXes33H777UycOJGlS5fy/PPPN3idhtS2xkQoeQn1nhpKZ2tJGNOwcAaIeUA3EekkItHAOOCd4AQiEjyZznnAmsDrj4D+IhIfaLAeSVDbxeEWUdlIHcYAEe71ILp06YLH4+Hhhx+uWtqz8gOxRYsWFBUVNdgLaNiwYcyZM4edO3fi9Xp54403qo4Fr8vwyiuvVO2vay2J5ORkmjZtWjV9+D/+8Y+q0kQoRowYwdSpU/H5fOTk5DB79uwG7yk4Lw3du60lYUzDwlbFpKoVIjIR92HvAaao6nIReQjIUtV3gIkichauh9JuXPUSqrpbRJ7CBRkFZqrq++HKa2U313COpr744ov5+uuvGTBgACJStR7EK6+8wuOPP05UVBSJiYm8+uqrbNmyheuuuw5/YALButaDuOaaa6rtGzt2LHfffTff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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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0YP+/ftz5MgRAHbt2kX//v3p06cPjz32GBERER7z+cwzz3DOOecwfPhwtm3bVrR98uTJ9OnThx49enD11VeTlZXFkiVLmDVrFn/729+Ii4tj165dHtMBhIWF0apVK37//fdKfW5KqRPzaSP16eTJbzex+eBxj/sycx0EBvgR5F+5eNm5aV0ev6zLCdONGzeOp556ilGjRrF+/XpuvfVWFi1aBMA999zD2LFjefPNNxk+fDi33HILTZs2BezNNy4urug4b7zxBoMGDSpx7MWLFxcFnOeff54333yTtWvXArBlyxY+//xzFi9eTGBgIHfffTdTpkyhS5cuHDhwgI0bNwKQmppKZGQkb775Ji+//DLx8fFlP6PMTPr3788zzzzDgw8+yOTJk3nkkUeYNGkSkyZNYvz48bz99tser3/VqlVMmzaNNWvW4HA46NWrF7179wbgqquu4o477gDgkUce4X//+x/33Xcfo0ePZtSoUYwZMwaAyMhIj+kA4uPjWbRoEX379j3h70Ip5b1aEyAqIkI5Y7yrRvfu3dmzZw9Tp04tamModNFFF7F7925++OEH5syZQ8+ePYtu3IVVTBU5dOgQ5Y0i/+WXX1i1ahV9+vQBIDs7m4YNG3LZZZexe/du7rvvPi699FIuvPDCE15DUFBQUbtJ7969mTt3LgBLly5l5syZAFx33XU88MADZd67aNEirrzySsLCwgAYPbp4WeeNGzfyyCOPkJqaSkZGBhdddJHH81eUrmHDhmzduvWE16CUqpxaEyAqetLffPA49UIDaFY/zGfnHz16NA888AALFiwgOTm5xL4GDRpw3XXXcd111zFq1CgWLlxY9IR9IqGhoeX22TfGcPPNN/Pcc8+V2bdu3Tp+/PFH3nrrLb744gvef//9Cs8TGBhY1F7j7++Pw1G5Xl/ldR2dMGECM2fOpEePHnz44YcsWLCg0ulycnIIDQ2tVH6UUidW69sgwJYgnD4sQYBtTH3sscfo1q1bie3z5s0rqk9PT09n165dtGjRwuvjdurUiZ07dxa9DgwMJD8/H4Bhw4bx1VdfcfSoHaSekpLC3r17SUpKwul0cvXVV/P000+zevVqAOrUqUN6enqlrqt///5Mnz4dgGnTpnlMM3jwYGbMmEF2djbp6el8++23RfvS09Np0qQJ+fn5TJkypWh76byUlw5g+/btdO3a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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "#opt=Adam(learning_rate=0.001)\n",
    "#modelf.compile(loss = LogCosh(), optimizer = 'adam', metrics = [\"mean_absolute_error\",\"mean_squared_error\"])\n",
    "modelf.compile(loss = MeanAbsoluteError(), optimizer = 'adam', metrics = [\"mean_squared_error\"])\n",
    "checkpoint = ModelCheckpoint(filepath='weights.{epoch:02d}.hdf5', monitor = 'loss', save_best_only=False, save_weights_only=True, period = 1)\n",
    "train_history= modelf.fit(Xtrain,Ytrain,batch_size = BATCH_SIZE, epochs = 100, callbacks = [checkpoint],validation_data=(Xval,Yval),initial_epoch=0)\n",
    "filepath = 'trainedmodel.csv'\n",
    "save_model(modelf, filepath)\n",
    "\n",
    "figure(0)\n",
    "plt.plot(train_history.history['loss'], label='loss (testing data)')\n",
    "plt.plot(train_history.history['val_loss'], label='loss (validation data)')\n",
    "plt.title('loss')\n",
    "plt.ylabel('logcosh loss')\n",
    "plt.xlabel('No. epoch')\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.savefig('coshloss.png')\n",
    "\n",
    "figure(1)\n",
    "plt.plot(train_history.history['mean_squared_error'], label='MSE (testing data)')\n",
    "plt.plot(train_history.history['val_mean_squared_error'], label='MSE (validation data)')\n",
    "plt.title('MSE Error')\n",
    "plt.ylabel('MSE value')\n",
    "plt.xlabel('No. epoch')\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.savefig('mseloss')\n",
    "'''\n",
    "figure(2)\n",
    "plt.plot(train_history.history['mean_absolute_error'], label='logcosh(testing data)')\n",
    "plt.plot(train_history.history['val_mean_absolute_error'], label='logcosh (validation data)')\n",
    "plt.title('MAE error')\n",
    "plt.ylabel('MAE value')\n",
    "plt.xlabel('No. epoch')\n",
    "plt.legend(loc=\"upper left\")\n",
    "plt.savefig('maeloss')\n",
    "'''\n",
    "plt.show()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_5\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "lstm_13 (LSTM)               (17, 24, 48)              11904     \n",
      "_________________________________________________________________\n",
      "dropout_13 (Dropout)         (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "lstm_14 (LSTM)               (17, 24, 48)              18624     \n",
      "_________________________________________________________________\n",
      "dropout_14 (Dropout)         (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "lstm_15 (LSTM)               (17, 24, 48)              18624     \n",
      "_________________________________________________________________\n",
      "dropout_15 (Dropout)         (17, 24, 48)              0         \n",
      "_________________________________________________________________\n",
      "time_distributed_5 (TimeDist (17, 24, 1)               49        \n",
      "=================================================================\n",
      "Total params: 49,201\n",
      "Trainable params: 49,201\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "cannot reshape array of size 14144 into shape (1088,24)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-27-85f8a2d2652b>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[0mmodelp\u001b[0m\u001b[1;33m=\u001b[0m \u001b[0mbuild_model\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mmodelp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msummary\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m \u001b[0mYval\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mYval\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1088\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m24\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m \u001b[0mYtrain\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0mYtrain\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m3417\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m24\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[0mYval\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0mscaler2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minverse_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mYval\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: cannot reshape array of size 14144 into shape (1088,24)"
     ]
    }
   ],
   "source": [
    "#modelp=load_model('trainedmodel.csv')\n",
    "\n",
    "\n",
    "modelp= build_model()\n",
    "modelp.summary()\n",
    "Yval=Yval.reshape(1088,24)\n",
    "Ytrain =Ytrain.reshape(3417,24)\n",
    "Yval =scaler2.inverse_transform(Yval)\n",
    "Ytrain = scaler2.inverse_transform(Ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_loss(epochs):\n",
    " valloss=[] ; trainloss=[]\n",
    " for i in range(10,epochs):\n",
    "     weightloc=\"weights.%d.hdf5\"%(i+1)\n",
    "     modelp.load_weights(weightloc)\n",
    "     Yvalpredicted =modelp.predict(Xval,batch_size=BATCH_SIZE)\n",
    "     Yvalpredicted =Yvalpredicted.reshape(1088,13)\n",
    "     \n",
    "     Yvalpredicted=scaler2.inverse_transform(Yvalpredicted)\n",
    "     valloss.append(sum(abs(Yval-Yvalpredicted)/(Yval+0.001))/(1088*13))\n",
    "     Ytrainpredicted =modelp.predict(Xtrain,batch_size=BATCH_SIZE)\n",
    "     Ytrainpredicted =Ytrainpredicted.reshape(3417,13)\n",
    "     \n",
    "     Ytrainpredicted = scaler2.inverse_transform(Ytrainpredicted)\n",
    "     trainloss.append(sum(abs((Ytrain-Ytrainpredicted)/(Ytrain+0.001)))/(3417*13))\n",
    " \n",
    " print(valloss,trainloss)\n",
    " figure(0)\n",
    " plt.plot(range(len(trainloss)),trainloss,label='trainloss')\n",
    " plt.plot(range(len(valloss)),valloss,label='valloss')\n",
    " plt.legend()\n",
    " plt.show()\n",
    " return(min(valloss),valloss.index(min(valloss)),min(trainloss),trainloss.index(min(trainloss)))\n",
    "\n",
    "     "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.9856677107859241, 0.9872258743773396, 0.9882713456373364, 0.9859214275740557, 0.9871157672913574, 0.9869668960647814, 0.9860780795644251, 0.9859199360071735, 0.9854042474082522, 0.9863168928106666, 0.9871970015549939, 0.9873435819005302, 0.986818713072891, 0.9851755221916793, 0.9869132766435998, 0.9864953846198545, 0.9853798164537955, 0.9868486370478412, 0.987528790396036, 0.9866494111343799, 0.9869295692454177, 0.9868683083014764, 0.9860646783343217, 0.9864244220242322, 0.9861455368217047, 0.986329737262789, 0.9865173581192352, 0.9859037441125096, 0.9883278045021023, 0.9872742681531135, 0.9867636125371609, 0.9872971812849229, 0.9873441986172687, 0.9863250552748151, 0.9880428764185243, 0.9867512512834611, 0.9865340644702807, 0.9885507315566296, 0.9861587682750296, 0.9886080289789226, 0.9860847540395533, 0.9868946073048629, 0.9873367215400761, 0.9860846191807764, 0.9874385485705964, 0.9861746581653876, 0.9897228817933238, 0.9875489737447979, 0.9864219207086242, 0.9854492452539431, 0.9868831949669501, 0.9881953766151658, 0.9866978491198853, 0.985469942860059, 0.9864336534336765, 0.9866060868147621, 0.9862690784433334, 0.987249445662239, 0.9868797662040875, 0.9868900468154028, 0.9875790107582237, 0.9863085433404836, 0.9890139230600323, 0.9884294201894175, 0.9857289090448852, 0.9862605672284124, 0.9864464380126972, 0.986987955379277, 0.9877135814115809, 0.9873902701192893, 0.9870062759729704, 0.9868784514472982, 0.9879114000081588, 0.9876655112938181, 0.9874856446699742, 0.9877213725230782, 0.9879775240812746, 0.9865876831473358, 0.9873547948331629, 0.9880138755548815, 0.9871234276044388, 0.9876810886075218, 0.9859705781338616, 0.9879517478530406, 0.9865182132332891, 0.9890612139585749, 0.9870639020600327, 0.9873598909440252, 0.9880307714747808, 0.9880684964201643] [0.9857830465508668, 0.9872537440304618, 0.988296699578376, 0.9860461572014487, 0.9871574315723217, 0.9870282247116998, 0.9861670361844294, 0.986036422835264, 0.9855533623889823, 0.9864543496136783, 0.9873013248084664, 0.987451550808566, 0.9869653323438378, 0.9853101373375783, 0.9871122075242993, 0.9866038607050778, 0.9855238468943807, 0.9869765449564056, 0.9876689816526107, 0.9867942192265209, 0.9870622661148706, 0.9869561459898003, 0.9861378974509516, 0.9865454833952202, 0.9862559157992903, 0.9864850911606982, 0.9866689288888976, 0.986106642570548, 0.9883637589943189, 0.9873600860763179, 0.9868613538478479, 0.9873724639227751, 0.9874121472753473, 0.9864440307501369, 0.9881441487036019, 0.9868250787807028, 0.9866577530763967, 0.9886367680920177, 0.9862525792908015, 0.9886657745949374, 0.9862212044284425, 0.9869482249815356, 0.9873851115272578, 0.9861960428119111, 0.9875579278295878, 0.9862381391584816, 0.9897500663768587, 0.9876520731976511, 0.986530799159252, 0.9856061465999109, 0.9869908002709692, 0.988272378965641, 0.9868460633961502, 0.985605896744983, 0.9865494005485834, 0.986766922012925, 0.9863703572746134, 0.9873166513743554, 0.9869963449781837, 0.986967787273825, 0.9877164341228138, 0.9864187869302673, 0.9891011603392365, 0.9884849450655327, 0.9858854100286836, 0.9863942769939599, 0.9865862124537235, 0.9870788099793106, 0.9878230479896839, 0.9874854759253905, 0.9871371806094731, 0.9869808350075077, 0.9880176059975683, 0.9877578479849886, 0.9875624015557769, 0.9878191646315911, 0.9880685520646135, 0.9867088114381983, 0.9874401480543498, 0.9880520407711796, 0.987233165546419, 0.9877756101187338, 0.9860583448696114, 0.9880277842508567, 0.9866410106734108, 0.9891361201471482, 0.987114711648691, 0.987433220057072, 0.9881256952641346, 0.9881024581516732]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "(0.9851755221916793, 13, 0.9853101373375783, 13)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "get_loss(100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "ename": "OSError",
     "evalue": "Unable to open file (unable to open file: name = 'weights.34.droput=0.2_3 layersof48_error=5%.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mOSError\u001b[0m                                   Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-24-d2562b322623>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mmodelp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mload_weights\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"weights.34.droput=0.2_3 layersof48_error=5%.hdf5\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\keras\\engine\\saving.py\u001b[0m in \u001b[0;36mload_wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    490\u001b[0m                 \u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mremove\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtmp_filepath\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    491\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0mres\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 492\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0mload_function\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m*\u001b[0m\u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    493\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    494\u001b[0m     \u001b[1;32mreturn\u001b[0m \u001b[0mload_wrapper\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\keras\\engine\\network.py\u001b[0m in \u001b[0;36mload_weights\u001b[1;34m(self, filepath, by_name, skip_mismatch, reshape)\u001b[0m\n\u001b[0;32m   1219\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mh5py\u001b[0m \u001b[1;32mis\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1220\u001b[0m             \u001b[1;32mraise\u001b[0m \u001b[0mImportError\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'`load_weights` requires h5py.'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1221\u001b[1;33m         \u001b[1;32mwith\u001b[0m \u001b[0mh5py\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mFile\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'r'\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1222\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[1;34m'layer_names'\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mattrs\u001b[0m \u001b[1;32mand\u001b[0m \u001b[1;34m'model_weights'\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1223\u001b[0m                 \u001b[0mf\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m'model_weights'\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\h5py\\_hl\\files.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, **kwds)\u001b[0m\n\u001b[0;32m    406\u001b[0m                 fid = make_fid(name, mode, userblock_size,\n\u001b[0;32m    407\u001b[0m                                \u001b[0mfapl\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfcpl\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mmake_fcpl\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrack_order\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtrack_order\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 408\u001b[1;33m                                swmr=swmr)\n\u001b[0m\u001b[0;32m    409\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    410\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlibver\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtuple\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\h5py\\_hl\\files.py\u001b[0m in \u001b[0;36mmake_fid\u001b[1;34m(name, mode, userblock_size, fapl, fcpl, swmr)\u001b[0m\n\u001b[0;32m    171\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mswmr\u001b[0m \u001b[1;32mand\u001b[0m \u001b[0mswmr_support\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    172\u001b[0m             \u001b[0mflags\u001b[0m \u001b[1;33m|=\u001b[0m \u001b[0mh5f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mACC_SWMR_READ\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 173\u001b[1;33m         \u001b[0mfid\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mh5f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mflags\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfapl\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mfapl\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    174\u001b[0m     \u001b[1;32melif\u001b[0m \u001b[0mmode\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m'r+'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    175\u001b[0m         \u001b[0mfid\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mh5f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mname\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mh5f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mACC_RDWR\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfapl\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mfapl\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mh5py\\_objects.pyx\u001b[0m in \u001b[0;36mh5py._objects.with_phil.wrapper\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mh5py\\_objects.pyx\u001b[0m in \u001b[0;36mh5py._objects.with_phil.wrapper\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mh5py\\h5f.pyx\u001b[0m in \u001b[0;36mh5py.h5f.open\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mOSError\u001b[0m: Unable to open file (unable to open file: name = 'weights.34.droput=0.2_3 layersof48_error=5%.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)"
     ]
    }
   ],
   "source": [
    "modelp.load_weights(\"weights.34.droput=0.2_3 layersof48_error=5%.hdf5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From D:\\New folder\\lib\\site-packages\\keras\\backend\\tensorflow_backend.py:422: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.\n",
      "\n"
     ]
    },
    {
     "ename": "InvalidArgumentError",
     "evalue": "Incompatible shapes: [32,192] vs. [17,192]\n\t [[{{node lstm_4/while/add_1}}]]",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mInvalidArgumentError\u001b[0m                      Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-18-a8b3dbfe5262>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mYvalpredicted\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0mmodelp\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mpredict\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mXval\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      2\u001b[0m \u001b[0mYvalpredicted\u001b[0m \u001b[1;33m=\u001b[0m\u001b[0mYvalpredicted\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1098\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m \u001b[0mYval\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mYval\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;36m1098\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m13\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\keras\\engine\\training.py\u001b[0m in \u001b[0;36mpredict\u001b[1;34m(self, x, batch_size, verbose, steps, callbacks, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[0;32m   1460\u001b[0m                                             \u001b[0mverbose\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mverbose\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1461\u001b[0m                                             \u001b[0msteps\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0msteps\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1462\u001b[1;33m                                             callbacks=callbacks)\n\u001b[0m\u001b[0;32m   1463\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1464\u001b[0m     def train_on_batch(self, x, y,\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\keras\\engine\\training_arrays.py\u001b[0m in \u001b[0;36mpredict_loop\u001b[1;34m(model, f, ins, batch_size, verbose, steps, callbacks)\u001b[0m\n\u001b[0;32m    322\u001b[0m             \u001b[0mbatch_logs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;33m{\u001b[0m\u001b[1;34m'batch'\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mbatch_index\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'size'\u001b[0m\u001b[1;33m:\u001b[0m \u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch_ids\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    323\u001b[0m             \u001b[0mcallbacks\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call_batch_hook\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'predict'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'begin'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbatch_index\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbatch_logs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 324\u001b[1;33m             \u001b[0mbatch_outs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mf\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mins_batch\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    325\u001b[0m             \u001b[0mbatch_outs\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mto_list\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mbatch_outs\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    326\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mbatch_index\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;36m0\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\tensorflow\\python\\keras\\backend.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, inputs)\u001b[0m\n\u001b[0;32m   3290\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3291\u001b[0m     fetched = self._callable_fn(*array_vals,\n\u001b[1;32m-> 3292\u001b[1;33m                                 run_metadata=self.run_metadata)\n\u001b[0m\u001b[0;32m   3293\u001b[0m     \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_call_fetch_callbacks\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfetched\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[0mlen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_fetches\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   3294\u001b[0m     output_structure = nest.pack_sequence_as(\n",
      "\u001b[1;32mD:\\New folder\\lib\\site-packages\\tensorflow\\python\\client\\session.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   1456\u001b[0m         ret = tf_session.TF_SessionRunCallable(self._session._session,\n\u001b[0;32m   1457\u001b[0m                                                \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_handle\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0margs\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1458\u001b[1;33m                                                run_metadata_ptr)\n\u001b[0m\u001b[0;32m   1459\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mrun_metadata\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1460\u001b[0m           \u001b[0mproto_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf_session\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTF_GetBuffer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrun_metadata_ptr\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mInvalidArgumentError\u001b[0m: Incompatible shapes: [32,192] vs. [17,192]\n\t [[{{node lstm_4/while/add_1}}]]"
     ]
    }
   ],
   "source": [
    "Yvalpredicted =modelp.predict(Xval)\n",
    "Yvalpredicted =Yvalpredicted.reshape(1098,13)\n",
    "Yval=Yval.reshape(1098,13)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [],
   "source": [
    "Yvalpredicted =scaler2.inverse_transform(Yvalpredicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "figure(3)\n",
    "plt.plot(range(6,19),Yval[400],'-bD',label='actual')\n",
    "plt.plot(range(6,19),Yvalpredicted[400],'-gD',label='predicted')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "metadata": {},
   "outputs": [],
   "source": [
    "Xval=scaler1.inverse_transform(Xval.reshape(-1, Xval.shape[-1])).reshape(Xval.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 2.45000000e+01  3.65000000e+01  8.04000000e+01  2.01700000e+03\n",
      "   1.11022302e-16  1.00000000e+00  1.00000000e+00 -2.44929360e-16\n",
      "   0.00000000e+00  1.00000000e+00]\n",
      " [ 2.49000000e+01  3.67000000e+01  7.91000000e+01  2.01700000e+03\n",
      "  -2.58819045e-01  9.65925826e-01  1.00000000e+00 -2.44929360e-16\n",
      "   1.90000000e+01  2.90000000e+01]\n",
      " [ 2.54000000e+01  3.68000000e+01  7.78000000e+01  2.01700000e+03\n",
      "  -5.00000000e-01  8.66025404e-01  1.00000000e+00 -2.44929360e-16\n",
      "   1.01000000e+02  1.06000000e+02]\n",
      " [ 2.58000000e+01  3.72000000e+01  7.59000000e+01  2.01700000e+03\n",
      "  -7.07106781e-01  7.07106781e-01  1.00000000e+00 -2.44929360e-16\n",
      "   4.24000000e+02  2.30000000e+02]\n",
      " [ 2.63000000e+01  3.77000000e+01  7.34000000e+01  2.01700000e+03\n",
      "  -8.66025404e-01  5.00000000e-01  1.00000000e+00 -2.44929360e-16\n",
      "   3.79000000e+02  4.48000000e+02]\n",
      " [ 2.68000000e+01  3.82000000e+01  7.11000000e+01  2.01700000e+03\n",
      "  -9.65925826e-01  2.58819045e-01  1.00000000e+00 -2.44929360e-16\n",
      "   2.65000000e+02  7.12000000e+02]\n",
      " [ 2.70000000e+01  3.87000000e+01  7.04000000e+01  2.01700000e+03\n",
      "  -1.00000000e+00  1.22464680e-16  1.00000000e+00 -2.44929360e-16\n",
      "   2.89000000e+02  8.38000000e+02]\n",
      " [ 2.69000000e+01  3.92000000e+01  7.14000000e+01  2.01700000e+03\n",
      "  -9.65925826e-01 -2.58819045e-01  1.00000000e+00 -2.44929360e-16\n",
      "   3.87000000e+02  7.08000000e+02]\n",
      " [ 2.68000000e+01  3.97000000e+01  7.25000000e+01  2.01700000e+03\n",
      "  -8.66025404e-01 -5.00000000e-01  1.00000000e+00 -2.44929360e-16\n",
      "   4.99000000e+02  6.10000000e+02]]\n"
     ]
    }
   ],
   "source": [
    "print(Xval[400,0:9,:])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4.711697851336954\n"
     ]
    }
   ],
   "source": [
    "print(sum(abs(Yval-Yvalpredicted)/(Yval+0.001))/(1098*13))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [],
   "source": [
    "Ytrainpredicted =modelp.predict(Xtrain)\n",
    "Ytrainpredicted =Ytrainpredicted.reshape(3417,13)\n",
    "Ytrain =Ytrain.reshape(3417,13)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "metadata": {},
   "outputs": [],
   "source": [
    "Ytrain = scaler2.transform(Ytrain)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'scaler2' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-f985757eae52>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mYtrainpredicted\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mscaler2\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minverse_transform\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mYtrainpredicted\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m: name 'scaler2' is not defined"
     ]
    }
   ],
   "source": [
    "Ytrainpredicted = scaler2.inverse_transform(Ytrainpredicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "figure(3)\n",
    "plt.plot(range(6,19),Ytrain[5],'-bD',label='actual')\n",
    "plt.plot(range(6,19),Ytrainpredicted[5],'-gD',label='predicted')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6.072624007612111\n"
     ]
    }
   ],
   "source": [
    "print(sum(abs((Ytrain-Ytrainpredicted)/(Ytrain+0.001)))/(3417*13))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 385,
   "metadata": {},
   "outputs": [],
   "source": [
    "Xtrain=scaler1.inverse_transform(Xval.reshape(-1, Xval.shape[-1])).reshape(Xval.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 388,
   "metadata": {},
   "outputs": [
    {
     "ename": "IndexError",
     "evalue": "index 995 is out of bounds for axis 0 with size 732",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-388-e9a2b733a4f5>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mXtrain\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;36m995\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;36m15\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mIndexError\u001b[0m: index 995 is out of bounds for axis 0 with size 732"
     ]
    }
   ],
   "source": [
    "print(Xtrain[995,2:15,:])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
