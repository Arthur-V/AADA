# Reconnaissance de forme : Compte-rendu

## TP 2 : Premier Réseau Convolutionnel

### Exemple d'un modèle de régression linéaire

> Question 1 : Analyser précisément cet exemple afin de bien comprendre son fonctionnement. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

>> Le réseau de neurone conçu est un modèle de régression linéaire. Il s’agit d’un problème d’apprentissage supervisé, car les données sont classées selon plusieurs catégories. Les données proviennent du dataset Iris. On cherche à déterminer les paramètres de la régression qui permet de classer chaque fleur dans la catégorie 2 ou au contraire en dehors de cette catégorie. Les vecteurs ```X``` et ```y``` sont de taille 150x2 dans la question 1. ```theta``` correspond aux paramètres de la régression linéaire et est de taille 2 (```theta[0]``` correspond a l’ordonnée à l’origine et ```theta[1]``` à la pente).

> Question 2 : Modifier le code afin de produire un modèle de regression linéaire sur des données 2D et 3D.

>> Le modèle actuel est un modèle 2D, ```theta``` est une variable qui possède 2 composantes. On peut faire un modèle sur des données de dimension supérieure, jusqu'à la dimension 5 (car chaque fleur du dataset Iris possède 4 attributs et une classe). Pour cela, on doit changer la dimension de ```X``` et ```theta```. Dans le code suivant, on fait la régression 5D. Le coefficient utilisé dans le calcul du gradient doit aussi être modifié. On constate une meilleure convergence si le coefficient choisi est égal à la dimension considérée.

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
