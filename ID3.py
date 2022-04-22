import pandas as pd
import math
import numpy as np

data = pd.read_csv("dataSet.csv")
features = [feat for feat in data]
dataBase = "Jugar Tenis"
values = np.unique(data[dataBase])
posi = values[0]
features.remove(dataBase)

# Clase nodo para guardar los valores e imprimir el arbol de decisión
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# Método para calcular la entropía
def calcEntropy(examples):
    pos = 0.0
    neg = 0.0
    for _, row in examples.iterrows():
        if row[dataBase] == posi:
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        p = pos / (pos + neg)
        n = neg / (pos + neg)
        return -(p * math.log(p, 2) + n * math.log(n, 2))

# Método para calcular la ganancia
def calcGain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = calcEntropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = calcEntropy(subdata)
        gain -= (float(len(subdata)) / float(len(examples))) * sub_e
    return gain

# Algoritmo ID3
def ID3(examples, attrs):
    # Se inicializa el nodo root (El primer nodo)
    root = Node()
    max_gain = 0
    max_feat = ""
    # Definir la ganancia mayor entre los atributos
    for feature in attrs:
        gain = calcGain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    # Definimos el valor del nodo root
    root.value = max_feat
    # Obtenemos los valores unicos de la columna con mayor ganancia
    uniq = np.unique(examples[max_feat])
    #Iteramos en estos valores
    for u in uniq:
        # Se crean los subset de datos en base a los valores unicos de la columna
        subdata = examples[examples[max_feat] == u]
        # Si el valor de la entropia es 0, significa que estamos en un nodo hoja
        if calcEntropy(subdata) == 0.0:
            newNode = Node()
            newNode.isLeaf = True
            newNode.value = u
            newNode.pred = np.unique(subdata[dataBase])
            root.children.append(newNode)
        # De otra forma, debemos crear un nuevo nodo con los nuevos datos
        else:
            dummyNode = Node()
            dummyNode.value = u
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)
            # Llamado del método para volver a iterar en este nodo
            child = ID3(subdata, new_attrs)
            # Se añade como hijo del nodo hijo a la llamada recursiva del método
            dummyNode.children.append(child)
            # Se añade al nodo inicial el nodo hijo
            root.children.append(dummyNode)
    return root

# Método para imprimir el árbol de decisión
def printTree(root: Node, depth=0):
    # Imprimir tabs para la visualización gráfica de la profundidad
    for i in range(depth):
        print("\t", end="")
    print(root.value, end="")
    # Si el nodo es una hoja, quiere decir que no tiene hijos, por lo que la impresión termina aquí
    if root.isLeaf:
        print(" -> ", root.pred)
    print()
    # Si el nodo tiene hijos, se hace la impresión de los mismos
    for child in root.children:
        printTree(child, depth + 1)

# Llamado al método
root = ID3(data, features)
printTree(root)
