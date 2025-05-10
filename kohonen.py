# coding: utf8
#!/usr/bin/env python
# ------------------------------------------------------------------------
# Carte de Kohonen
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------
# Implémentation de l'algorithme des cartes auto-organisatrices de Kohonen
# ------------------------------------------------------------------------
# Pour que les divisions soient toutes réelles (pas de division entière)
from __future__ import division
# Librairie de calcul matriciel
import numpy
# Librairie d'affichage
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm



class Neuron:
  ''' Classe représentant un neurone '''
  
  def __init__(self, w, posx, posy):
    '''
    @summary: Création d'un neurone
    @param w: poids du neurone
    @type w: numpy array
    @param posx: position en x du neurone dans la carte
    @type posx: int
    @param posy: position en y du neurone dans la carte
    @type posy: int
    '''
    # Initialisation des poids
    self.weights = w.flatten()
    # Initialisation de la position
    self.posx = posx
    self.posy = posy
    # Initialisation de la sortie du neurone
    self.y = 0.
  
  def compute(self,x):
    '''
    @summary: Affecte à y la valeur de sortie du neurone (ici on choisit la distance entre son poids et l'entrée, i.e. une fonction d'aggrégation identité)
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # DONE
    self.y = numpy.sqrt(numpy.sum((self.weights-x)**2))

  def learn(self,eta,sigma,posxjetoile,posyjetoile,x):
    '''
    @summary: Modifie les poids selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param posxjetoile: position en x du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posxjetoile: int
    @param posyjetoile: position en y du neurone gagnant (i.e. celui dont le poids est le plus proche de l'entrée)
    @type posyjetoile: int
    @param x: entrée du neurone
    @type x: numpy array
    '''
    # DONE (attention à ne pas changer la partie à gauche du =)
    self.weights[:] = self.weights + eta * numpy.exp(-((self.posx-posxjetoile)**2+(self.posy-posyjetoile)**2)/(2*sigma**2)) * (x - self.weights)

class SOM:
  ''' Classe implémentant une carte de Kohonen. '''

  def __init__(self, inputsize, gridsize):
    '''
    @summary: Création du réseau
    @param inputsize: taille de l'entrée
    @type inputsize: tuple
    @param gridsize: taille de la carte
    @type gridsize: tuple
    '''
    # Initialisation de la taille de l'entrée
    self.inputsize = inputsize
    # Initialisation de la taille de la carte
    self.gridsize = gridsize
    # Création de la carte
    # Carte de neurones
    self.map = []    
    # Carte des poids
    self.weightsmap = []
    # Carte des activités
    self.activitymap = []
    for posx in range(gridsize[0]):
      mline = []
      wmline = []
      amline = []
      for posy in range(gridsize[1]):
        neuron = Neuron(numpy.random.random(self.inputsize),posx,posy)
        mline.append(neuron)
        wmline.append(neuron.weights)
        amline.append(neuron.y)
      self.map.append(mline)
      self.weightsmap.append(wmline)
      self.activitymap.append(amline)
    self.activitymap = numpy.array(self.activitymap)

  def compute(self,x):
    '''
    @summary: calcule de l'activité des neurones de la carte
    @param x: entrée de la carte (identique pour chaque neurone)
    @type x: numpy array
    '''
    # On demande à chaque neurone de calculer son activité et on met à jour la carte d'activité de la carte
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].compute(x)
        self.activitymap[posx][posy] = self.map[posx][posy].y

  def learn(self,eta,sigma,x):
    '''
    @summary: Modifie les poids de la carte selon la règle de Kohonen
    @param eta: taux d'apprentissage
    @type eta: float
    @param sigma: largeur du voisinage
    @type sigma: float
    @param x: entrée de la carte
    @type x: numpy array
    '''
    # Calcul du neurone vainqueur
    jetoilex,jetoiley = numpy.unravel_index(numpy.argmin(self.activitymap),self.gridsize)
    # Mise à jour des poids de chaque neurone
    for posx in range(self.gridsize[0]):
      for posy in range(self.gridsize[1]):
        self.map[posx][posy].learn(eta,sigma,jetoilex,jetoiley,x)

        
      

  def scatter_plot(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée (utilisable dans le cas d'entrée à deux dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Modification des limites de l'affichage
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def scatter_plot_2(self,interactive=False):
    '''
    @summary: Affichage du réseau dans l'espace d'entrée en 2 fois 2d (utilisable dans le cas d'entrée à quatre dimensions et d'une carte avec une topologie de grille carrée)
    @param interactive: Indique si l'affichage se fait en mode interactif
    @type interactive: boolean
    '''
    # Création de la figure
    if not interactive:
      plt.figure()
    # Affichage des 2 premières dimensions dans le plan
    plt.subplot(1,2,1)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,0].flatten(),w[:,:,1].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,0],w[i,:,1],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,0],w[:,i,1],'k',linewidth=1.)
    # Affichage des 2 dernières dimensions dans le plan
    plt.subplot(1,2,2)
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Affichage des poids
    plt.scatter(w[:,:,2].flatten(),w[:,:,3].flatten(),c='k')
    # Affichage de la grille
    for i in range(w.shape[0]):
      plt.plot(w[i,:,2],w[i,:,3],'k',linewidth=1.)
    for i in range(w.shape[1]):
      plt.plot(w[:,i,2],w[:,i,3],'k',linewidth=1.)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace d\'entree')
    # Affichage de la figure
    if not interactive:
      plt.show()

  def plot(self):
    '''
    @summary: Affichage des poids du réseau (matrice des poids)
    '''
    # Récupération des poids
    w = numpy.array(self.weightsmap)
    # Création de la figure
    f,a = plt.subplots(w.shape[0],w.shape[1])    
    # Affichage des poids dans un sous graphique (suivant sa position de la SOM)
    for i in range(w.shape[0]):
      for j in range(w.shape[1]):
        plt.subplot(w.shape[0],w.shape[1],i*w.shape[1]+j+1)
        im = plt.imshow(w[i,j].reshape(self.inputsize),interpolation='nearest',vmin=numpy.min(w),vmax=numpy.max(w),cmap='binary')
        plt.xticks([])
        plt.yticks([])
    # Affichage de l'échelle
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    f.colorbar(im, cax=cbar_ax)
    # Affichage du titre de la figure
    plt.suptitle('Poids dans l\'espace de la carte')
    # Affichage de la figure
    plt.show()

  def quantification(self,X):
    '''
    @summary: Calcul de l'erreur de quantification vectorielle moyenne du réseau sur le jeu de données
    @param X: le jeu de données
    @type X: numpy array
    '''
    # On récupère le nombre d'exemples
    nsamples = X.shape[0]
    # Somme des erreurs quadratiques
    s = 0
    # Pour tous les exemples du jeu de test
    for x in X:
      # On calcule la distance à chaque poids de neurone
      self.compute(x.flatten())
      # On rajoute la distance minimale au carré à la somme
      s += numpy.min(self.activitymap)**2
    # On renvoie l'erreur de quantification vectorielle moyenne
    return s/nsamples
  
  def organisation(self):
    '''
    @summary: Calcul un indice en fonction de l'organisation des poids en fonction des poids et des coordonnées.
    '''
    # On calcule la liste du ratio entre la differance de poids et la distance entre chaque paire de neurones
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

def aff_graph(X,Y1,Y2 = None,title="",xlabel="X"):
  if Y2 is not None:
    # plt.figure()
    # plt.plot(X,Y1)
    # plt.plot(X,Y2)
    # plt.title(title)
    # plt.legend(['Quantification','Organisation'])
    # plt.show()
    # plt.figure()

    fig, ax1 = plt.subplots()
    # Axe pour la première courbe (Quantification)
    ax1.plot(X, Y1, 'b-')  # Courbe en bleu
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Quantification', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Création d'un second axe pour la deuxième courbe (Organisation)
    ax2 = ax1.twinx()
    ax2.plot(X, Y2, 'r-')  # Courbe en rouge
    ax2.set_ylabel('Organisation', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Titre et affichage
    plt.title(title)
    fig.tight_layout()  # Ajuste les marges pour éviter les chevauchements
    plt.show()
  else:
    plt.figure()
    plt.plot(X,Y1)
    plt.title(title)
    plt.show()

def run_single(args):
    eta, sigma, N, samples = args
    return run_kohonen_2D(eta, sigma, N, samples)

def run_kohonen_2D_avg(eta,sigma,N,samples,nb_runs=12):

  quantification_list = []
  organisation_list = []

  args = [(eta, sigma, N, samples) for _ in range(nb_runs)]

  with ProcessPoolExecutor(max_workers=nb_runs) as executor:
    results = list(executor.map(run_single, args))

  for quantification, organisation in results:
    quantification_list.append(quantification)
    organisation_list.append(organisation)

  return numpy.mean(quantification_list), numpy.mean(organisation_list)

def run_kohonen_2D(eta,sigma,N,samples,affichage=False,network=None):
  if network is None:
    network = SOM((2,1),(10,10))
  nsamples = samples.shape[0]
  if affichage:
    plt.ion()
  for i in range(N+1):

    if i%10==0:
      eta = eta * 0.995
      sigma = sigma * 0.995
    index = numpy.random.randint(nsamples)
    x = samples[index].flatten()
    network.compute(x)
    network.learn(eta,sigma,x)
    if i%100==0 and affichage:
      plt.clf()
      network.scatter_plot(True)
      plt.pause(0.00001)
      plt.draw()
  if affichage:
    print("Quantification : ",network.quantification(samples))
    print("Organisation : ",network.organisation())
    plt.close()
    plt.ioff()
    network.scatter_plot(False)
    plt.draw()
  return network.quantification(samples),network.organisation()

def get_samples(num_samples,affichage=False):
  nsamples = 1200
  if num_samples==1:
    samples = numpy.random.random((nsamples,2,1))*2-1
  
  elif (num_samples==2):
    # Ensemble de données deux carrés
    samples1 = numpy.random.random((nsamples//2,2,1))
    samples1[:,0,:] -= 1
    samples2 = numpy.random.random((nsamples//2,2,1))
    samples2[:,1,:] -= 1
    samples = numpy.concatenate((samples1,samples2))
  
  elif (num_samples==3):
    samples1 = numpy.random.random((nsamples//3,2,1))*2-1
    samples2 = numpy.random.random((nsamples//3,2,1))*2-1
    samples3 = numpy.random.random((nsamples//3,2,1))*2-1
    samples2[:,0,:] /= 6
    samples3[:,1,:] /= 6
    samples = numpy.concatenate((samples1,samples2,samples3))
  
  elif (num_samples==4):
    samples1 = numpy.random.normal(0,2,(nsamples - nsamples//4,2,1))/8
    samples2 = numpy.random.random((nsamples//4,2,1))*2-1
    samples = numpy.concatenate((samples1,samples2))

  elif (num_samples==9):
    # Ensemble de données robotiques
    samples = numpy.random.random((nsamples,4,1))
    samples[:,0:2,:] *= numpy.pi
    l1 = 0.7
    l2 = 0.3
    samples[:,2,:] = l1*numpy.cos(samples[:,0,:])+l2*numpy.cos(samples[:,0,:]+samples[:,1,:])
    samples[:,3,:] = l1*numpy.sin(samples[:,0,:])+l2*numpy.sin(samples[:,0,:]+samples[:,1,:])
  else:
    print("Erreur : nombre d'échantillons non reconnu")
    return None
  if affichage:
    if num_samples!=9:
      plt.figure()
      plt.scatter(samples[:,0,0], samples[:,1,0])
      plt.xlim(-1,1)
      plt.ylim(-1,1)
      plt.suptitle('Donnees apprentissage')
      plt.show()
    else:
      plt.figure()
      plt.subplot(1,2,1)
      plt.scatter(samples[:,0,0].flatten(),samples[:,1,0].flatten(),c='k')
      plt.subplot(1,2,2)
      plt.scatter(samples[:,2,0].flatten(),samples[:,3,0].flatten(),c='k')
      plt.suptitle('Donnees apprentissage')
      plt.show()

  return samples

# -----------------------------------------------------------------------------

def main_test():

  eta = 1 # Taux d'apprentissage
  sigma = 2 # Largeur du voisinage
  n = 3000 # Nombre de pas de temps d'apprentissage

  num_samples = 1

  quantification_list = []
  organisation_list = []

  iterateur = numpy.arange(0.3,1.8,0.1)
  xlab = "eta initial"
  for eta in tqdm(iterateur):
    samples = get_samples(num_samples)
    (quantification,organisation) = run_kohonen_2D_avg(eta,sigma,n,samples)
    quantification_list.append(quantification)
    organisation_list.append(organisation)

  aff_graph(iterateur,quantification_list,organisation_list,xlabel=xlab)

def main_test_rapide():

  eta = 1 # Taux d'apprentissage
  sigma = 2 # Largeur du voisinage
  n = 3000 # Nombre de pas de temps d'apprentissage
  network = SOM((2,1),(10,10))

  num_samples = 1
  samples = get_samples(num_samples,affichage=True)
  (quantification,organisation) = run_kohonen_2D(eta,sigma,n,samples,affichage=True,network=network)

def main_test_nb_iter():

  eta = 1 # Taux d'apprentissage
  sigma = 2 # Largeur du voisinage
  n = 5000 # Nombre de pas de temps d'apprentissage

  num_samples = 1
  affichage = False

  if True:
    samples = get_samples(num_samples,affichage=affichage)
    network = SOM((2,1),(10,10))
    nsamples = samples.shape[0]
    test_sur_n_ordonnée = []
    test_sur_n_quantification = []
    test_sur_n_organisation = []
    if affichage:
      plt.ion()
    for i in tqdm(range(n+1)):
      if i%10==0:
        eta = eta * 0.995
        sigma = sigma * 0.995
      index = numpy.random.randint(nsamples)
      x = samples[index].flatten()
      network.compute(x)
      network.learn(eta,sigma,x)
      if i%250==0 and affichage:
        network.scatter_plot(False)
      if i%100==0 and i >= 200:
        test_sur_n_ordonnée.append(i)
        test_sur_n_quantification.append(network.quantification(samples))
        test_sur_n_organisation.append(network.organisation())
    if affichage:
      network.scatter_plot(False)
      plt.ioff()

  aff_graph(test_sur_n_ordonnée,test_sur_n_quantification,test_sur_n_organisation,xlabel="i")
  print("Quantification : ",network.quantification(samples))
  print("Organisation : ",network.organisation())

if __name__ == '__main__':
  
  # main_test()
  # main_test_rapide()
  # main_test_nb_iter()
  print("FINI !")