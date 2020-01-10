# Reconnaissance de forme : Compte-rendu

## TP 2 : Premier Réseau Convolutionnel

### Exemple d'un modèle de régression linéaire

> Question 1 : Analyser précisément cet exemple afin de bien comprendre son fonctionnement. Attacher une importance particulière à la structuration des tableaux de données qui servent à l'apprentissage.

>> Le réseau de neurone conçu est un modèle de régression linéaire. Il s’agit d’un problème d’apprentissage supervisé, car les données sont classées selon plusieurs catégories. Les données proviennent du dataset Iris. On cherche à déterminer les paramètres de la régression qui permet de classer chaque fleur dans la catégorie 2 ou au contraire en dehors de cette catégorie. Les vecteurs ```X``` et ```y``` sont de taille 150x2 dans la question 1. ```theta``` correspond aux paramètres de la régression linéaire et est de taille 2 (```theta[0]``` correspond a l’ordonnée à l’origine et ```theta[1]``` à la pente).
