# Carte Auto-organisatrice de Kohonen (SOM)

Ce dépôt contient une implémentation complète en Python de l'algorithme des **cartes auto-organisatrices de Kohonen** (Self-Organizing Maps). Ce projet a été réalisé dans le cadre du module d'Intelligence Artificielle et Réseaux de Neurones.

## Fonctionnalités du Code

L'implémentation repose sur une architecture orientée objet flexible:

* **Classe `Neuron`** : Gère les poids individuels, le calcul de la distance à l'entrée et la règle de mise à jour de Kohonen.
* **Classe `SOM`** : Gère la grille de neurones, la détermination du neurone gagnant (BMU - Best Matching Unit) et l'apprentissage global sur la carte.
* **Métriques d'évaluation** :
    * **Erreur de quantification vectorielle** : Mesure la qualité de la représentation des données par les poids.
    * **Indice d'organisation** : Calcul basé sur le ratio entre la différence de poids et la distance topologique pour évaluer la qualité de l'auto-organisation.
* **Outils de visualisation** : Affichage des poids dans l'espace d'entrée (2D et 4D) et visualisation de la matrice des poids sous forme de niveaux de gris.
* **Optimisation** : Utilisation intensive de `numpy` pour les calculs matriciels et support du multi-processing via `ProcessPoolExecutor` pour accélérer les tests statistiques[cite: 94, 107].

## Dépendances

Le projet nécessite les bibliothèques suivantes :
* `numpy` (calcul numérique) 
* `matplotlib` (visualisation) 
* `tqdm` (barres de progression) 

```bash
pip install numpy matplotlib tqdm
```

## Utilisation

Le fichier principal est `kohonen.py`. Il contient plusieurs fonctions de test prédéfinies dans le bloc `if __name__ == '__main__':`. Pour lancer une expérimentation, décommentez la fonction souhaitée à la fin du fichier:

1.  **`main_test()`** : Génère des graphiques de performance (quantification vs organisation) en faisant varier le taux d'apprentissage $\eta$.
2.  **`main_test_rapide()`** : Lance un apprentissage simple sur des données uniformes avec un affichage interactif en temps réel.
3.  **`main_test_nb_iter()`** : Analyse l'évolution des métriques en fonction du nombre d'itérations d'apprentissage.
4.  **`main_bras_robotique()`** : Apprentissage sur un jeu de données à 4 dimensions représentant les angles et la position d'un bras robotique[cite: 114, 115, 107].

Lancer le script :
```bash
python kohonen.py
```

## Structure du projet

* `kohonen.py` : Cœur de l'algorithme et scripts de test.
* `img/` : Dossier contenant les images et graphiques générés pour le rapport.

---
*Auteur : Thomas Blanché* 
