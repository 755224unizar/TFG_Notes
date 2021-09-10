"""
* AUTOR : Daniel Tosaus Lanza
* NIA: 755224
* FICHERO : eval.py
* FECHA : Mayo de 2021
* DESCRIPCION: Código que evalúa un modelo de red con un conjunto de datos de test que se le proporcionan. Tras dar las probabilidades a
               posteriori la red de cada clase para cada ejemplo se usan estos datos para generar una curva ROC de cada una de las clases,
               siendo el área encerrada bajo ellas un parámetro de precisión que se puede usar para comparar modelos.
"""
# Definicion del modelo
import numpy as np
import torch.nn.functional as F
import torch
import statistics
import sys
print("Defino Modelo")
sys.stdout.flush()
class CardioNet2(torch.nn.Module):
  def __init__(self, factor, ini):
    super(CardioNet2, self).__init__()
    self.conv1 = torch.nn.Conv2d(15,ini, kernel_size=5, padding=2, stride=(1, 4))
    self.batchNC1 = torch.nn.BatchNorm2d(ini)
    self.conv2 = torch.nn.Conv2d(ini,int(factor*ini), kernel_size=5, padding=2, stride=(1, 4))
    self.batchNC2 = torch.nn.BatchNorm2d(int(factor*ini))
    self.conv3 = torch.nn.Conv2d(int(factor*ini),int((factor**2)*ini), kernel_size=3, padding=1, stride=(2, 4))
    self.batchNC3 = torch.nn.BatchNorm2d(int((factor**2)*ini))
    #self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
    #self.batchNC4 = torch.nn.BatchNorm2d(256)
    self.reshape_dim = int((factor**2)*ini)
    self.drop = torch.nn.Dropout(0.5)

    self.lin1 = torch.nn.Linear(int((factor**2)*ini*12), int((factor**8)*ini), bias=True)
    self.batchNL1 = torch.nn.BatchNorm1d(int((factor**8)*ini))
    self.lin2 = torch.nn.Linear(int((factor**8)*ini), 4, bias=True)
    self.batchNL2 = torch.nn.BatchNorm1d(4)
 
  def forward(self, x):
    B = x.shape[0]
    x = F.relu(self.batchNC1(self.conv1(x)))
    #print(x.shape)
    x = F.relu(self.batchNC2(self.conv2(x)))
    #print(x.shape)
    x = F.relu(self.batchNC3(self.conv3(x)))
    #x = F.relu(self.batchNC4(self.conv4(x)))
    #print(x.shape)
    x = torch.mean(x, 2)                # Elimino la dimensión de las ventanas con un average pooling
    x = self.drop(x)
    #print(x.shape)
    x = torch.reshape(x, (B, self.reshape_dim*12))  # Elimino la última dimensión con un reshape
    #print(x.shape)
    x = F.relu(self.batchNL1(self.lin1(x)))    
    #print(x.shape)
    x = self.batchNL2(self.lin2(x))
    
    #x = torch.sigmoid(x)    # Previa a la sigmoide
    return x
  def predict(self, x):
    B = x.shape[0]
    x = F.relu(self.batchNC1(self.conv1(x)))
    x = F.relu(self.batchNC2(self.conv2(x)))
    x = F.relu(self.batchNC3(self.conv3(x)))
    #x = F.relu(self.batchNC4(self.conv4(x)))
    x = torch.mean(x, 2)                # Elimino la dimensión de las ventanas con un average pooling
    x = self.drop(x)
    x = torch.reshape(x, (B, self.reshape_dim*12))  # Elimino la última dimensión con un reshape
    x = F.relu(self.batchNL1(self.lin1(x)))    
    x = self.batchNL2(self.lin2(x))
    x = torch.sigmoid(x)
    return x


print("Cargo Modelo")
sys.stdout.flush()
model = CardioNet2(2.5,16)
# Carga del modelo entrenado
path = '/home/alumnos/alumno5/work/Modelos/modelo5_canales/modelo5_4_20ep.pt'
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
pytorch_total_params = sum(p.numel() for p in model.parameters())

# Carga de datos de test y validacion
print("Cargo Datos")
sys.stdout.flush()

import pickle
with open('/home/alumnos/alumno5/work/Modelos/modelo5_canales/test.txt', 'rb') as f:
    x_test = pickle.load(f)
#with open('/home/alumnos/alumno5/work/Modelos/modelo5_canales/validacion.txt', 'rb') as f:
    #x_validacion = pickle.load(f)

# Necesito definir el metodo que forma los tensores
import scipy.io
import os
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
tabla = {164889003:0, 426783006:1, 270492004:2, 59118001:3}
def convertCatToLabel(y):
  #labels = np.zeros((1, 111), dtype=np.int64)
  labels = np.zeros((1, 4), dtype=np.int64)
  for i in range(y.shape[1]):
    
    if y[0,i] in tabla.keys():                #Si la patología es una de las 27 pongo a 1 la misma en el vector de 27 componentes
      labels[0,tabla[int(y[0,i])]]=1
    
  return labels
class ECG_SubDataset(Dataset):
 
  def __init__(self, datos):
  
    path = '/scratch/alumno5/work/datos_entreno/SubsetEtiquetasMultiples'

    self.files_list = datos

    x_list = []
    y_list = []
      
    self.x = x_list
    self.y = y_list
    self.n_samples = len(datos)
    
 
  def __getitem__(self, index):

    ok = False
    #print("Index: ", index)
    #sys.stdout.flush()
    path = '/scratch/alumno5/work/datos_entreno/SubsetEtiquetasMultiples'
    while not ok:
      xy = scipy.io.loadmat(path+'/'+self.files_list[index])
      #print(self.files_list[index])
      matrices = xy['tensor']
      diagnostico = (xy['diagnostico'])
      
      x_n = matrices        
      y_n = convertCatToLabel(diagnostico)

      
      n = 10;                               # numero de ventanas temporales a tomar
      t = x_n.shape[1]                      # Tengo que asegurar que la segunda dimension del tensor (ventanas) es mayor que 0
      if t>0:
        ok = True
      else:
        index = np.random.randint(0, self.n_samples)



    #rep = (n/t)
    rep = (t/n)
    #print(t, rep)
    
    if rep>=1:                            # Si tengo suficientes ventanas las cojo
      if n==t:
        ind= 0
      else:
        ind = np.random.randint(0, t-n)   # Podríamos haber cogido un trozo de la señal especialmente significativo, en vez de aleatorio
      
      #print("Cojo tramo de "+str(ind)+" hasta "+str(ind+n))
      return x_n[:, ind:ind+n, :], y_n
 
    else:                                 # Si no hay suficientes tengo que concatenar datos repetidos
      #print("Hay que concatenar")
      res = x_n
      #print("Tamaño inicial: "+ str(res.shape[1]))
      while (res.shape[1]<=n):
        #print("Entro al while")
        res = np.concatenate((x_n, res), axis=1)
        #print("NUEVO Tamaño tensor: "+ str(res.shape[1]))
      #print(res.shape[1])
      #print(n)
      ind = np.random.randint(0, res.shape[1]-n)
      return res[:, ind:ind+n,:] , y_n
 
 
    
 
  def __len__(self):
    return self.n_samples

dataset_test = ECG_SubDataset(x_test)
#dataset_vali = ECG_SubDataset(x_validacion)
B = 32
dataloader_test = DataLoader(dataset=dataset_test, batch_size=B, shuffle=False, num_workers=0)
#dataloader_vali = DataLoader(dataset=dataset_vali, batch_size=B, shuffle=False, num_workers=0)
 
dataIterator_test = iter(dataloader_test)
#dataIterator_vali = iter(dataloader_vali)

print("Datos de test: ", len(x_test))
#print("Datos de val: ", len(x_validacion))
sys.stdout.flush()

y_test = torch.zeros([len(x_test), 4], dtype=torch.int64)
test_probs = torch.zeros([len(x_test), 4], dtype=torch.float32)

#y_val = torch.zeros([len(x_validacion), 4], dtype=torch.int64)
#val_probs = torch.zeros([len(x_validacion), 4], dtype=torch.float32)
# Evaluacion del modelo
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
model.eval()
print("Empiezo evaluacion del modelo")
sys.stdout.flush()

cont = 0

for features, labels in dataIterator_test:
    prob = model.predict(features.to(device))
    test_probs[cont*B:cont*B+B, :] = prob.cpu()
    y_test[cont*B:cont*B+B,:] = labels.squeeze().cpu()
    cont = cont + 1


# Generacion de arrays np de probabilidades para cada clase

np_test_pr = test_probs.cpu().detach().numpy()        #numpy arrays que contienen las probabilidades a posteriori de cada clase para cada fichero
np_y_test = y_test.cpu().numpy()                      # Seran de dimensiones len(test) x 4 y len(val) x 4

#np_val_pr = val_probs.cpu().detach().numpy()          

# En este punto tenemos las probabilidades a posteriori y las etiquetas para todos los ficheros de test y validacion

# Definicion de la funcion de calculo de area bajo curvas ROC
print("Calculo de AUCs y ROCs")
sys.stdout.flush()
def auc(scores, labels, plot=False, plt_name='example'):

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels.ravel(), scores.ravel())
    roc_auc = auc(fpr, tpr)    
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(fpr, tpr, color='r', lw=2, label='ROC (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic '+plt_name)
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig('/home/alumnos/alumno5/work/Modelos/modelo5_canales/2.5_16/curva_roc'+plt_name+'.png')
    return roc_auc

# Calculo de AUCs y ROCs
test_auc0 = auc(np_test_pr[:,0], np_y_test[:,0], plot=True, plt_name='AF')
test_auc1 = auc(np_test_pr[:,1], np_y_test[:,1], plot=True, plt_name='SNR')
test_auc2 = auc(np_test_pr[:,2], np_y_test[:,2], plot=True, plt_name='IAVB')
test_auc3 = auc(np_test_pr[:,3], np_y_test[:,3], plot=True, plt_name='RBBB')
print("AUC de clase AF: ", test_auc0)
print("AUC de clase SNR: ", test_auc1)
print("AUC de clase IAVB: ", test_auc2)
print("AUC de clase RBBB: ", test_auc3)
print("AUC media del modelo: ", statistics.mean([test_auc0, test_auc1, test_auc2, test_auc3]))
print("Numero de Parametros del Modelo: ", pytorch_total_params)
