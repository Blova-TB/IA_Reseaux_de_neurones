# 3.1

- si le taux d'apprentissage est de 0, la valeur des poids du neuronne gagnant ne changera pas car $\Delta w _{ij} = \eta * [...]$

- si le taux d'apprentissage est 1, $ \Delta w _{ij} = 1 * e^{0} * (x_{i} - w_{ji}) = (x_{i} - w_{ji}) $ on modifie donc le poid du neuronne gagnant de la difference entre ses poid est celui de l'entré. les poid du neuronne gagnant deviennent donc celui de l'entré.

- le nouveau poids sera entre le poid initial et le poid de l'entré $ w _{ji} = w _{ji} + \eta(x_{i} - w_{ji}) = (1-\eta) w_{ji} + \eta x_{i}$

- si $ \eta > 1$ les poids du neuronne gagnant depasse l'entré. ils passe de l'autre côté de l'entre. En quel que sorte le neuronne a "trop appris" ...

Formule :

$ \Delta w _{ij} = \eta * e^{-\frac{||j-j^{*}||^{2}_{c}}{2 \sigma ^{2}}} (x_{i} - w_{ji})$
