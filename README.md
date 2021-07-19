Predicting the solar irradiance, helps to guess the PV power output and greatly
increase the efficiency of the smart electric grid.
This plays a major role in shifting towards PV production and reducing the carbon
footprint. An deep learning model was trained for this purpose using solar irradiance and
meteorological data of last 14 years
Correlations were found out between the irradiance and different meteorological
parameters like temperature, humidity, air pressure etc. to observe on which variables, the
irradiance depended strongly on, and an LSTM with 24-time steps was modeled to predict
the day-ahead hourly solar irradiance. Upon training and arriving at the best values of the hyper-parameters, the final model was
able to predict day-ahead solar irradiance with a 6% error. This could have been further
decreased, if I had access to the cloud-type data for training the model,which I couldn't
obtain
