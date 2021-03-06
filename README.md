# M.Sc. Thesis: Applitcation to LSTM and MLP to modeling structral dynamic responses and damage assessment
## Abstract
This study presents a neural network-based approach for the dynamic response modeling and damage detection of linear and nonlinear structural systems under earthquake motion. The proposed approach consists of two steps. The first step is to develop data-driven one-step ahead prediction models of a healthy structure via multi-layer perceptron (MLP) and long short-term memory (LSTM) neural network by utilizing the base excitations and dynamic responses of the structure under some seismic events. These well-trained models can accurately predict the dynamic responses of the structure under different seismic events, which are not considered in establishing the models. The second step is to perform damage diagnosis of the structure at the current stage by investigating the dynamic prediction errors from the established neural network models. The considerable differences between the predicted and measured responses indicate that damage occurred in the building, and several error-based indices are proposed herein to enhance the damage assessment. The feasibility of the proposed approach is demonstrated through numerically simulated responses of a linear and nonlinear eight-floor shear buildings subjected to recorded earthquakes, and different damage scenarios by deducing the stiffness of one or two stories are investigated. This work also studies the effects of the number of training seismic events, noise corruption, and incomplete measurements on the performance of the prediction from the trained neural networks. It is shown that LSTM has an advantage over MLP in learning temporal relationships.

In addition, this work also employs LSTM and MLP to predict base excitations of a structure from its dynamic responses and predict the elastic modulus of each story in a building from its stiffness matrix via MLP. The promising results from numerical simulation data indicate that the proposed methods are potentially suitable for real applications.

## Instructions
Filenames instruction:
* "linear" file: linear system's one-step-ahead time series forecasting models and damage assessment
* "nonlinear model" file: nonlinear system(Bouc Wen model)'s one-step-ahead time series forecasting models and damage assessment
* "predict_unmeasured_floor" file: predict unmeasured floors from measured dynamic responses and base excitations
* "future_predict_past" file: predict base excitations of a structure from its dynamic responses
* "K_E_mapping" file: predict the elastic modulus of each story in a building from its stiffness matrix

For downloading training and test dataset used in the above files please refer to the following link: [Dataset](https://drive.google.com/drive/folders/1wgOAK0811hNiYYuOg-M8BNqgLSfM7Ve6?usp=sharing)
