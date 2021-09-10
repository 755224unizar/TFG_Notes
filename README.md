# CollabNotebookTFG
TFG Notes about: Implementation and evaluation of a neural network based system to diagnose cardiac pathologies from ECG signals.
Including all the versions of the modules that are used in the final version of the signal processing and the neural network, and the design decisions made.
This document shows the development of the project from the beginning, exposing the problems that arised and the solutions implemented for them.
# Matlab Processing
This folder contains all the Matlab code used for this project, being most of it self-programmed (the exceptions are: wavedet.m, readheader.m and baseline2.m).
In DesarrolloCodigoTFG.m you can find previous versions of the code used in the final version, and see the development of the code between versions.
The ProcesadoEnSerie.m file covers all the processing made to the ECG signals across the datasets used to take the ECG signals as input and serve 15 images with its characteristics as the output
# Network Models, Train, Test and Evaluation Code and Data
In this folder we can find the models of the neural networks used in the project, as well as the examples used for their training, test, and evaluation.
It also contains the eval.py code, used for the evaluation of the models with concurrent classes through the computing of the ROC curves and their AUC.
