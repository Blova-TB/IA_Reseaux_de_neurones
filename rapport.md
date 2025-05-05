# Intelligence Artificielle Reseaux de neurones

## 3 Étude "théorique" de cas simples (7 points)

### 3.1 Influence de $\eta$

- si le taux d'apprentissage est de 0, la valeur des poids du neurone gagnant ne changera pas car $\Delta w _{ij} = \eta * [...]$

- si le taux d'apprentissage est 1, $ \Delta w _{ij} = 1 * e^{0} * (x_{i} - w_{ji}) = (x_{i} - w_{ji}) $ on modifie donc le poid du neurone gagnant de la difference entre ses poid est celui de l'entré. les poid du neurone gagnant deviennent donc celui de l'entré.

- le nouveau poids sera entre le poid initial et le poid de l'entré $ w _{ji} = w _{ji} + \eta(x_{i} - w_{ji}) = (1-\eta) w_{ji} + \eta x_{i}$

- si $ \eta > 1$ les poids du neurone gagnant depasse l'entré. ils passe de l'autre côté de l'entre. En quel que sorte le neurone a "trop appris" ...

### 3.2 Influence de $\sigma$

- si $ \sigma$ augmente, $\frac{||j-j^{*}||^{2}_{c}}{2 \sigma ^{2}}$ va diminuer et donc $e^{-\frac{||j-j^{*}||^{2}_{c}}{2 \sigma ^{2}}}$ va se raprocher de 1. L'influance de la distance sur $\Delta w _{ij}$ va donc diminuer. les neurones plus eloigné vont donc plus apprendre.

- beaucoup plus "resserrée" puisque tout les neurones seront affecté a chaque nouvelle entré.

-   
  - SOLUTION 1 : La variances des distances des poids des neurones par rapport au centre (moyenne des poids de tout les neurones). Avec $n$ le nombre de neurones et $x$ un vecteur de meme taille que $w_{j}$ tel que $x_{i}=\frac{1}{n}\sum_{j=1}^{n} w_{ji}$ --> mesure $=\frac{1}{n}\sum^{n}{(||x - w_{j}||^{2})}$
  - SOLUTION 2 : Apres reflexion, il me parrait plus judicieux de se baser sur la rapport entre la distance entre les neurones selon leurs coordonnés et selon leur poids. Pour ensuite calculer la variance de ces rapports. On calcule donc la moyenne avec $C_{j}$ (ou $C_{i}$) les coordonnées du neurones numero j (ou i) : 
  $Moy = \frac{n(n+1)}{2} \sum_{i=1}^{n}{\sum_{j=i+1}^{n}{\frac{d(W_{j},W_{i})}{d(C_{j},C_{i})}}}$
  $ Variance = \frac{n(n+1)}{2} \sum_{i=1}^{n}{\sum_{j=i+1}^{n}{(\frac{d(W_{j},W_{i})}{d(C_{j},C_{i})} - Moy)^{2}}}$

### 3.3 Influence de la distribution d'entrée

- deux entrées $X_1$ et $X_2$
  - les poids du neurone vont se stabiliser entre les poids de $X_1$ et $X_2$
  - les poids du neurone vont se stabiliser sur le segment entre $X_1$ et $X_2$ en ettant n fois plus proche de $X_1$.
  - les poids du neurone vont tendre vers la moyennes des poids contenus dans la BDD
- si le neurone est le neurone gagnant, il vas etre attiré par l'entrée. Sinon, cela vas dependre de sa distance avec le neurone gagant selon leurs coordonnés et de la diferance entre ses poids et ceux de l'entrée. Donc si deux neurone voisin on des poids tres different, lorsqu'un des deux sera le neurone gagant, l'autre sera enormement attiré par l'entré et les deux se rapprocheront fortement.
- les neurones vont donc se reaprtire de facon similaire a leur repartition selon leur coordonées.

## 4 Étude pratique

### 4.2 Implémentation

```python
def compute(self,x):
  self.y = numpy.sqrt(numpy.sum((self.weights-x)**2))

def learn(self,eta,sigma,posxjetoile,posyjetoile,x):
  self.weights[:] = self.weights + eta * numpy.exp(-((self.posx-posxjetoile)**2+(self.posy-posyjetoile)**2)/(2*sigma**2)) * (x - self.weights)

def organisation(self):
  ratio = []
  for posx1 in range(self.gridsize[0]):
    for posy1 in range(self.gridsize[1]):
      for posx2 in range(self.gridsize[0]):
        for posy2 in range(self.gridsize[1]):
          if(posx1 != posx2 or posy1 != posy2):
            d = numpy.sqrt(numpy.sum((self.map[posx1][posy1].weights - self.map[posx2][posy2].weights)**2))
            d2 = numpy.sqrt((posx1 - posx2)**2+(posy1 - posy2)**2)
            ratio.append(d/d2)
  return numpy.var(ratio)
```

### 4.3 Analyse de l'algorithme




Formule :

$ \Delta w _{ij} = \eta * e^{-\frac{||j-j^{*}||^{2}_{c}}{2 \sigma ^{2}}} (x_{i} - w_{ji})$
