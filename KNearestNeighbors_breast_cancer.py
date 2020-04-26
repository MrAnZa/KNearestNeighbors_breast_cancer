######### LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
from sklearn import datasets
########## PREPARAR LA DATA ##########
#Importamos los datos de la misma librería de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)
########## ENTENDIMIENTO DE LA DATA ##########
#Verifico la información contenida en el dataset
print('Información en el dataset:')
print(dataset.keys())
print()
#Verifico las características del dataset
print('Características del dataset:')
print(dataset.DESCR)
#Seleccionamos todas las columnas
X = dataset.data
#Defino los datos correspondientes a las etiquetas
y = dataset.target
########## IMPLEMENTACIÓN DE K Vecinos mas cercanos ##########
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Defino algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier

algoritmo= KNeighborsClassifier(n_neighbors=7,metric='minkowski',p=2)
#Entreno el Modelo
algoritmo.fit(X_train,y_train)
#Realizo una Prediccion
y_predict=algoritmo.predict(X_test)
#Verificar matriz de confusión
from sklearn.metrics import confusion_matrix

matriz= confusion_matrix(y_test,y_predict)
print('Matriz de confusión:')
print(matriz)

#Calcular la precision del modelo
from sklearn.metrics import precision_score

precision=precision_score(y_test,y_predict)
print('Precisión del modelo')
print(precision)
