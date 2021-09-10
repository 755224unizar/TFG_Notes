"""
* AUTOR : Daniel Tosaus Lanza
* NIA: 755224
* FICHERO : modelo3.py
* FECHA : Mayo de 2021
* DESCRIPCION: Modelo 0: modelo entrenado con el subset de 4 patologías ampliado, pero en este caso considerando que las clases 
               pueden ser concurrentes.
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import scipy.io
import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

# Definimos diccionario para transformar las etiquetas en indices
tabla = {164889003:0, 426783006:1, 270492004:2, 59118001:3}
 
def convertCatToLabel(y):
  #labels = np.zeros((1, 111), dtype=np.int64)
  labels = np.zeros((1, 4), dtype=np.int64)
  for i in range(y.shape[1]):
    #print("Etiqueta: ", y[0,i])
    if y[0,i] in tabla.keys():                #Si la patología es una de las 27 pongo a 1 la misma en el vector de 27 componentes
      labels[0,tabla[int(y[0,i])]]=1
    #labels[0,tabla_completa[int(y[0,i])]]=1
  return labels
 
# Definición del Dataloader
class ECG_SubDataset(Dataset):
 
  def __init__(self, datos):
    # data loading
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
    # Nos aseguramos de que las dimensiones del tensor cargado son correctas antes de continuar
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
 
 


# Segundo modelo sencillo de red para el subset de 4 patologías       Batch de dimensiones 32x15x10x750
import numpy as np
import torch.nn.functional as F
import torch
 
class CardioNet2(torch.nn.Module):
  def __init__(self):
    super(CardioNet2, self).__init__()
    self.conv1 = torch.nn.Conv2d(15,32, kernel_size=5, padding=2, stride=(1, 4))
    self.batchN1 = torch.nn.BatchNorm2d(32)
    self.conv2 = torch.nn.Conv2d(32,64, kernel_size=5, padding=2, stride=(1, 4))
    self.batchN2 = torch.nn.BatchNorm2d(64)
    self.conv3 = torch.nn.Conv2d(64,128, kernel_size=3, padding=1, stride=(2, 4))
    self.batchN3 = torch.nn.BatchNorm2d(128)

    self.drop = torch.nn.Dropout(0.5)

    self.lin1 = torch.nn.Linear(128*12, 512, bias=True)
    self.batchN4 = torch.nn.BatchNorm1d(512)
    self.lin2 = torch.nn.Linear(512, 4, bias=True)
    self.batchN5 = torch.nn.BatchNorm1d(4)
 
  def forward(self, x):
    B = x.shape[0]
    x = F.relu(self.batchN1(self.conv1(x)))
    #print(x.shape)
    x = F.relu(self.batchN2(self.conv2(x)))
    #print(x.shape)
    x = F.relu(self.batchN3(self.conv3(x)))
    #print(x.shape)
    x = torch.mean(x, 2)                # Elimino la dimensión de las ventanas con un average pooling
    x = self.drop(x)
    #print(x.shape)
    x = torch.reshape(x, (B, 128*12))  # Elimino la última dimensión con un reshape
    #print(x.shape)
    x = F.relu(self.batchN4(self.lin1(x)))    
    #print(x.shape)
    x = self.batchN5(self.lin2(x))
    
    #x = torch.sigmoid(x)    # Previa a la sigmoide
    return x
  def predict(self, x):
    B = x.shape[0]
    x = F.relu(self.batchN1(self.conv1(x)))
    x = F.relu(self.batchN2(self.conv2(x)))
    x = F.relu(self.batchN3(self.conv3(x)))
    x = torch.mean(x, 2)                # Elimino la dimensión de las ventanas con un average pooling
    x = self.drop(x)
    x = torch.reshape(x, (B, 128*12))  # Elimino la última dimensión con un reshape
    x = F.relu(self.batchN4(self.lin1(x)))    
    x = self.batchN5(self.lin2(x))
    x = torch.sigmoid(x)
    return x

# Declaración del modelo  y comprobación de las dimensiones con un tensor de datos aleatorios
model = CardioNet2()
J = torch.nn.BCEWithLogitsLoss()
np_tensor = np.random.random((32,15,10,750))
tensor_prueba = torch.tensor(np_tensor, dtype=torch.float32)
print(tensor_prueba.shape)
o = model(tensor_prueba)
print(o.shape)

# Separación de los datos en datos de train y de test
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Si es el primer entrenamiento separamos en train, test y validacion. Despues conservaremos los mismos para poder comparar
hay_ficheros = os.path.isfile('train.txt') and os.path.isfile('test.txt') and os.path.isfile('validacion.txt')
#hay_ficheros = True
if not hay_ficheros:
  files = []
  with open('/scratch/alumno5/work/datos_entreno/SubsetEtiquetasMultiples/NombresFicheros.txt', 'r') as texto:
    lines = texto.readlines()
    for l in lines:
      as_list = l.split("\n")
      files.append(as_list[0])

  datos, validacion = train_test_split(files, test_size = 0.1)        #Extraemos datos de validacion  
  x_train, x_test = train_test_split(datos, test_size = 0.2)          #Separamos en datos de train y test

  #Guardo datos de train, test y de validacion
  
  file = open('/home/alumnos/alumno5/work/Modelos/modelo3/validacion.txt', 'wb')
  pickle.dump(validacion, file)
  file.close()
  file = open('/home/alumnos/alumno5/work/Modelos/modelo3/test.txt', 'wb')
  pickle.dump(x_test, file)
  file.close()
  file = open('/home/alumnos/alumno5/work/Modelos/modelo3/train.txt', 'wb')
  pickle.dump(x_train, file)
  file.close()
  # Ejemplo de carga de variables guardadas
  #with open('x_train.txt', 'rb') as f:
  #    x_train = pickle.load(f)
else:
  with open('train.txt', 'rb') as f:
    x_train = pickle.load(f)
  with open('test.txt', 'rb') as f:
    x_test = pickle.load(f)
  with open('validacion.txt', 'rb') as f:
    validacion = pickle.load(f)


print("Train: ", len(x_train))
print("Test: ", len(x_test))

# Declaración del modelo, optimizador y funcion de coste para realizar un entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CardioNet2()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
J = torch.nn.BCEWithLogitsLoss()
dataset_train = ECG_SubDataset(x_train)
dataset_test = ECG_SubDataset(x_test)
print(device)

path = '/home/alumnos/alumno5/work/Modelos/modelo3/modelo3_20ep.pt'
# Se define el dataloader para entrenar la red, siendo los batches de 32 ficheros
dataloader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
dataloader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
dataIterator_train = iter(dataloader_train)
dataIterator_test = iter(dataloader_test)
precisiones_test = []

model.zero_grad()
model.to(device)

# Proceso de entrenamiento durante 20 epochs
epochs = 20      
costes = []
costes_epoch = []
for epoch in range(epochs):
  model.train()
  print('Epoch: '+ str(epoch))
  costes_epoch = []
  dataIterator_train = iter(dataloader_train)
  #print("Cargo datos Para el Epoch")
  for features, labels in dataIterator_train:
    #print("Datos Cargados")
    features = features.to(device)
    o = model(features)
    #print(o.shape)
    y = torch.reshape(labels,(32,4))
    y = torch.tensor(y, device = device, dtype=torch.float32)
    coste = J(y,o)
    model.zero_grad()
    coste.backward()
    optim.step()
    costes_epoch.append(coste.cpu())
    print(coste.cpu())
  coste_epoch = torch.mean(torch.tensor(costes_epoch))
  print(epoch, coste_epoch)
  sys.stdout.flush()
  costes.append(coste_epoch)
  # Guardamos el estado del modelo tras cada epoch de entrenamiento
  torch.save({
            #'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': coste,
            }, path)
  
  
# Representación gráfica de la evolución del coste durante el entrenamiento


plt.plot(costes)
plt.title('Coste por epoch')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Coste')
plt.savefig('/home/alumnos/alumno5/work/Modelos/modelo3/costes_20ep.png')
plt.close()

""""
plt.plot(precisiones_test)
plt.title('Precisión en test por epoch')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Precisión')
plt.savefig('/home/alumnos/alumno5/work/Modelos/modelo3/precisiones_test50epochs_4clases.png')
"""