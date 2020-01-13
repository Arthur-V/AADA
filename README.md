# Reconnaissance de forme : Compte-rendu

## TP 2 : Premier Réseau Convolutionnel

### Exemple d'un modèle de régression linéaire

> Question 1 : Analyser précisément cet exemple afin de bien comprendre son fonctionnement. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

>> Le réseau de neurone conçu est un modèle de régression linéaire. Il s’agit d’un problème d’apprentissage supervisé, car les données sont classées selon plusieurs catégories. Les données proviennent du dataset Iris. On cherche à déterminer les paramètres de la régression qui permet de classer chaque fleur dans la catégorie 2 ou au contraire en dehors de cette catégorie. Les vecteurs ```X``` et ```y``` sont de taille 150x2 dans la question 1. ```theta``` correspond aux paramètres de la régression linéaire et est de taille 2 (```theta[0]``` correspond a l’ordonnée à l’origine et ```theta[1]``` à la pente).

> Question 2 : Modifier le code afin de produire un modèle de regression linéaire sur des données 2D et 3D.

>> Le modèle actuel est un modèle 2D, en effet, ```theta``` est possède 2 composantes. On peut faire un modèle sur des données de dimension supérieure, jusqu'à la dimension 5 (car chaque fleur du dataset Iris possède 4 attributs et une classe, soit au plus 4 dimensions pour les attributs et une dimension supplémentaire pour la classe). Pour cela, on doit changer la dimension de ```X``` et ```theta```. Dans le code suivant, on fait la régression 5D. Le coefficient utilisé dans le calcul du gradient doit aussi être modifié. On constate une meilleure convergence si le coefficient choisi est égal à la dimension considérée.

```
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

iris = datasets.load_iris()
nbExemples = len(iris["data"])

# On créé une liste qui contient 1 lorsque la fleur est de type 2 et 0 sinon pour faire une classification
Type = (iris["target"] == 2).astype(np.int)
print("Type : ", Type)

# On va programmer une résolution par descente de gradients 'à la main'
# Paramètre de la descente de gradient
learning_rate = 0.01 
display_step = 1
n_epochs = 10000

# Initialisation des tenseurs de constantes
X = tf.constant(np.c_[np.ones((nbExemples,1)), iris["data"]],dtype=tf.float32, name = "X") #X est un tensor qui contient les features 'Largeur' et une colonne de '1' pour le Theta0
y = tf.constant(Type, shape = (nbExemples,1), dtype = tf.float32, name="y") # y est un tensor qui représente deux classes possibles

# Modèle
# theta est un tensor de 5 variables en colonne initialisées aléatoirement entre -1 et +1
theta = tf.Variable(tf.random_uniform([5,1], -1.0, 1.0),  name = "theta") 

# la prédicton est faite avec la fonction logistique, pred est le tensor de toutes les prédictions
pred = tf.sigmoid(tf.matmul(X,theta)) 

# l'error est le tensor de toutes les erreurs de prédictions
error = pred - y 

# calcule la MSE qui est en fait la valeur que minimise une descente de gradient sur une fonction logistique
mse = tf.reduce_mean(tf.square(error), name="mse") 

# Calcul du gradient de l'erreur
gradients = (5/nbExemples) * tf.matmul(tf.transpose(X), error) 

# Definition de la fonction de correction de theta à partir du gradient
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer() # créer un noeud init dans le graphe qui correspond à l'initialisation

# Execution du modèle
with tf.Session() as sess:
    # On Execute le noeud d'initialisation des variables
    sess.run(init) 
    for epoch in range(n_epochs):
    	# affichage tous les 100 pas de calcul
        if epoch % 1000 == 0:  
            print("Epoch", epoch, "MSE =", mse.eval())
        # Exécution d'un pas de recalcule de theta avec appels de tous les opérateurs et tensors nécessaires dans le graphe
        sess.run(training_op) 
    best_theta = theta.eval()
print("Best theta : ", best_theta)
```

### Convolutional Neural Network for handwritten digits recognition

> Question 3 : Bien analyser la structuration du script qui est assez standard et que vous retrouverez dans la suite des expérimentations. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

>> Le code a la même structure que le code du MLP. On commence par définir le réseau et les fonctions nécessaires à la phase d'apprentissage (fonction de Loss ```loss```, d'entraînement ```train``` - ici la méthode du gradient, précision ```accuracy```). On entraîne ensuite le modèle avec un nombre fixé d'itérations. A chaque itération, on entraîne le modèle sur tous les batchs grâce à la commande ```train```, puis on fait la prédiction du modèle sur les batchs d'entraînement et de test. Ensuite, on affiche l'erreur sur les batchs d'entraînement et de test en fonction du numéro de l'itération. On affiche enfin pour chacune des images de test la valeur retournée par le modèle et la valeur attendue.

Code de l'architecture du réseau convolutif :

```
taille_noyau = 5

nbr_noyau = 16
# Premiere couche
b01 = tf.constant(np.zeros(nbr_noyau), dtype=tf.float32)
w0 = tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(ph_images.get_shape()[-1]), nbr_noyau)))
couche_0 = tf.nn.relu(tf.nn.conv2d(ph_images, w0, strides=[1, 1, 1, 1], padding='SAME') + b01)
# Deuxième couche
w1 = tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_0.get_shape()[-1]), nbr_noyau)))
couche_1 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(couche_0, w1, strides=[1, 1, 1, 1], padding='SAME') + b01),
                          ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

nbr_noyau = 32
# Troisième couche
b23 = tf.constant(np.zeros(nbr_noyau), dtype=tf.float32)
w2 = tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_1.get_shape()[-1]), nbr_noyau)))
couche_2 = tf.nn.relu(tf.nn.conv2d(couche_1, w2, strides=[1, 1, 1, 1], padding='SAME') + b23)
# Quatrième couche
w3 = tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_2.get_shape()[-1]), nbr_noyau)))
couche_3 = tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(couche_2, w3, strides=[1, 1, 1, 1], padding='SAME') + b23),
                          ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
```

> Question 4 : que peut on dire des variables $b$ et des pondérations $w$ ?

>> Les variables $b$ sont les biais (dans le cadre de ce TP on les a fixé à 0). Les pondérations $w$ sont des vecteurs qui sont des paramètres du réseau. Chaque neurone est le résultat d'une opération (produit tensoriel par $w$ ou multiplication par $w$ - selon la dimension de la variable d'entrée du neurone - puis somme avec $b$). L'algorithme consiste à choisir les meilleurs valeurs de $w$ de manière 

Code de mise en place de l'applatissement de la dernière couche de convolution et création de deux couches *fully-connected* : 

```
sortie_CNN = tf.contrib.layers.flatten(couche_3)

nbr_in = sortie_CNN.get_shape()[-1]
nbr_outFC1 = 512
nbr_outFC2 = 10

wFC1=tf.Variable(tf.truncated_normal(shape=(nbr_in, nbr_outFC1)), dtype=tf.float32)
bFC1 = tf.constant(np.zeros(shape=(nbr_outFC1)), dtype=tf.float32)
FC1 = tf.nn.sigmoid(tf.matmul(sortie_CNN, wFC1) + bFC1)

wFC2=tf.Variable(tf.truncated_normal(shape=(nbr_outFC1, nbr_outFC2)), dtype=tf.float32)
bFC2 = tf.constant(np.zeros(shape=(nbr_outFC2)), dtype=tf.float32)
FC2 = tf.nn.sigmoid(tf.matmul(FC1, wFC2) + bFC2)
scso = tf.nn.softmax(FC2)
```

Code de définition de la fonction de Loss et des métriques de précision :

```
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=FC2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(FC2, 1), tf.argmax(ph_labels, 1)), tf.float32))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
```

> Question 5 : 
Créer les deux fonctions suivantes afin d'alléger le code.
```
def convolution(couche_prec, taille_noyau, nbr_noyau):

    return result

def fc(couche_prec, nbr_neurone):

    return result
```

```
def convolution(couche_prec, taille_noyau, nbr_noyau):
    b = tf.constant(np.zeros(nbr_noyau), dtype=tf.float32)
    w = tf.Variable(tf.random.truncated_normal(shape=(taille_noyau, taille_noyau, int(couche_prec.get_shape()[-1]), nbr_noyau)))
    return tf.nn.relu(tf.nn.conv2d(couche_prec, w, strides=[1, 1, 1, 1], padding='SAME') + b)

def fc(couche_prec, nbr_neurone):
    bFC = tf.constant(np.zeros(shape=(nbr_neurone)), dtype=tf.float32)
    wFC = tf.Variable(tf.truncated_normal(shape=(couche_prec.get_shape()[-1], nbr_neurone)), dtype=tf.float32)
    return tf.nn.sigmoid(tf.matmul(couche_prec, wFC) + bFC)
```

Le code qui permet de définir le réseau de neurones est le suivant :

```
taille_noyau = 5

couche_0 = convolution(ph_images, taille_noyau, 16)
couche_1 = convolution(couche_0, taille_noyau, 16)
couche_1 = tf.nn.max_pool(couche_1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

couche_2 = convolution(couche_1, taille_noyau, 32)
couche_3 = convolution(couche_2, taille_noyau, 32)
couche_3 = tf.nn.max_pool(couche_3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

flattened = tf.contrib.layers.flatten(couche_3)

FC1 = fc(flattened, 512)
FC2 = fc(FC1, 10)
scso = tf.nn.softmax(FC2)
```

On définit également deux fonctions qui permettent de simplifier la boucle principale :

```
def trainNetwork(taille_batch, train, mnist_train_images, mnist_train_labels):
    for batch in np.arange(0, len(mnist_train_images), taille_batch):
        s.run(train, feed_dict={
            ph_images: mnist_train_images[batch:batch+taille_batch],
            ph_labels: mnist_train_labels[batch:batch+taille_batch]
            })

def computeAccuracy(taille_batch, accuracy, images, labels):
    return np.mean(
            [s.run(accuracy, feed_dict={
                ph_images: images[batch:batch+taille_batch],
                ph_labels: labels[batch:batch+taille_batch]})
            for batch in range(0, len(images), taille_batch) ]
            )
```

Le code est le suivant :
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot
from google.colab.patches import cv2_imshow

# Data
mnist_train_images=np.fromfile("drive/My Drive/dataset/mnist/train-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_train_labels=np.eye(10)[np.fromfile("drive/My Drive/dataset/mnist/train-labels.idx1-ubyte", dtype=np.uint8)[8:]]
mnist_test_images=np.fromfile("drive/My Drive/dataset/mnist/t10k-images.idx3-ubyte", dtype=np.uint8)[16:].reshape(-1, 28, 28, 1)/255
mnist_test_labels=np.eye(10)[np.fromfile("drive/My Drive/dataset/mnist/t10k-labels.idx1-ubyte", dtype=np.uint8)[8:]]

# Parameters
taille_batch=100
nbr_entrainement=15
learning_rate=0.001
taille_noyau = 5

# Network definition
ph_images=tf.placeholder(shape=(None, 28, 28, 1), dtype=tf.float32)
ph_labels=tf.placeholder(shape=(None, 10), dtype=tf.float32)
couche_0 = convolution(ph_images, taille_noyau, 16)
couche_1 = convolution(couche_0, taille_noyau, 16)
couche_1 = tf.nn.max_pool(couche_1, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
couche_2 = convolution(couche_1, taille_noyau, 32)
couche_3 = convolution(couche_2, taille_noyau, 32)
couche_3 = tf.nn.max_pool(couche_3, ksize = [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
flattened = tf.contrib.layers.flatten(couche_3)
FC1 = fc(flattened, 512)
FC2 = fc(FC1, 10)
scso = tf.nn.softmax(FC2)

# Metrics definition
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=ph_labels, logits=FC2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(FC2, 1), tf.argmax(ph_labels, 1)), tf.float32))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    
    tab_train=[]
    tab_test=[]
    
    for id_entrainement in np.arange(nbr_entrainement):
        print("> Entrainement", id_entrainement)

        trainNetwork(taille_batch, train, mnist_train_images, mnist_train_labels)
        accuracy_train = computeAccuracy(taille_batch, accuracy,
                                         mnist_train_images, mnist_train_labels)
        accuracy_test = computeAccuracy(taille_batch, accuracy,
                                        mnist_test_images, mnist_test_labels)

        print("  train:", accuracy_train, "\n  test :", accuracy_test)
        tab_train.append(1 - accuracy_train)
        tab_test.append(1 - accuracy_test)

    plot.ylim(0, 1)
    plot.grid()
    plot.plot(tab_train, label="Train error")
    plot.plot(tab_test, label="Test error")
    plot.legend(loc="upper right")
    plot.show()
    
    resulat=s.run(scso, feed_dict={ph_images: mnist_test_images[0:taille_batch]})
    np.set_printoptions(formatter={'float': '{:0.3f}'.format})

    for image in range(taille_batch):
        print("image", image)
        print("sortie du réseau:", resulat[image], np.argmax(resulat[image]))
        print("sortie attendue :", mnist_test_labels[image], np.argmax(mnist_test_labels[image]))
        cv2_imshow(mnist_test_images[image]*255)
```

Le graphe suivant représente l'évolution de l'erreur sur les ensembles d'entraînenement et de test en fonction du nombre d'itérations.

![Evolution erreur premier réseau](images/Entrainement_CNN_sans_normalisation.png?raw=true "Evolution de l'erreur pour les ensemble d'entraînement et de test")