import numpy as np
from igraph import Graph
from igraph import ADJ_MAX

# ADD get_params, set_params to _BaseGraphClustering
# ADD new class that perfoms given clustering on a SET of graphs not a single graph


'''
import numpy as np
from networkx import karate_club_graph
from networkx import adjacency_matrix
from sklearn.metrics import adjusted_mutual_info_score

adjacency = adjacency_matrix(karate_club_graph()).todense()
true_club = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
                      1, 1, 2, 2, 1, 1, 2, 1, 2, 1, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

louvain = clustering_louvain()
newman = clustering_newman_eigenvector(n_clusters=2)
fgreedy = clustering_fgreedy(n_clusters=2)

labels_louvain = louvain.fit_transform(adjacency)
labels_newman = newman.fit_transform(adjacency)
labels_fgreedy = fgreedy.fit_transform(adjacency)

print('Louvain: {}\nNewman Eigenvector: {}\nFast Greedy: {}'.format(
    adjusted_mutual_info_score(true_club, labels_louvain),
    adjusted_mutual_info_score(true_club, labels_newman),
    adjusted_mutual_info_score(true_club, labels_fgreedy)))
    
    
>>> Louvain: 0.34930981296735714
>>> Newman Eigenvector: 0.8324015819025615
>>> Fast Greedy: 0.6699133628094176
'''

class _BaseGraphClustering():
    
    '''
    Basic graph clustering class,
    
    methods:
    
    internal method (start with _) preferably should not return anything
    and just store output in class variable (self.)
    
    out method should return the result
    
    fit ---> build_igraph/build_nxgraph ---> fit_igraph/fit_nxgraph
    variables:
    '''

    def __init__(self,  n_clusters=None, weight_attr='weight'):
        
        '''
        INPUTS : 
        
        n_clusters,
        weight_attr
        
        CLASS VARIABLES :
        
        _n_clusters -     inputed preferrable number of clusters
        n_clusters -      resulting number of clusters
        weight_attr -     name of edge weight attribute
        iGraph -          igraph Graph object
        nxGraph -         networkx Graph object
        n_steps -         resulting number of steps for clustering algorithm to converge
        levels -          intermediate clustering results 
        _adjacency_list - adjacency matrix as python list
        '''
        
        self._n_clusters = n_clusters
        self.n_clusters = None
        self.weight_attr = weight_attr
        self.iGraph = None
        self.nxGraph = None
        self.n_steps = None
        self.levels = None
        self._adjacency_list = None
        
    def _build_igraph(self, weighted=True):
        
        if weighted:
            self.iGraph = Graph.Weighted_Adjacency(
                self._adjacency_list,
                mode=ADJ_MAX,
                attr=self.weight_attr)
        else:
            pass

    def _build_nxgraph(self, weighted=True):
        pass
    
    def fit_transform(self, adjacency, weighted=True, backend='igraph', ):
        
        self._adjacency_list = adjacency.tolist()       

        if backend=='igraph':
            self._build_igraph(weighted)
            self._fit_igraph()
            self.n_clusters = np.unique(self.labels).shape[0]
            
        elif backend=='networkx':
            pass
        else:
            pass
        
        return self.labels
    
    def _fit_igraph(self, ):
        pass
    
    def _fit_nx(self, ):
        pass
    
class clustering_louvain(_BaseGraphClustering):
       
    def _fit_igraph(self, ):
        
        levels = self.iGraph.community_multilevel(
            weights=self.weight_attr,
            return_levels=True)
        
        self.levels = np.array([level.membership for level in levels]).astype(int)
        self.n_steps = len(levels)
        self.labels = np.array(self.levels[-1])
        
    def _fit_nx():
        pass


class clustering_newman_eigenvector(_BaseGraphClustering):
    
    def _fit_igraph(self, ):
        
        self.labels = self.iGraph.community_leading_eigenvector(
            clusters=self._n_clusters,
            weights=self.weight_attr).membership
        self.labels = np.array(self.labels)
        
    def _fit_nx():
        pass

class clustering_fgreedy(_BaseGraphClustering):
    
    '''
    during initialization n_clusters=None 
    is preferable, it return optimal number of clusters,
    if _n_clusters = int > 0 return label vector
    with this number of clusters
    '''
    
    def _fit_igraph(self, ):
        
        self.levels = self.iGraph.community_fastgreedy(
                weights=self.weight_attr)
        
        if self._n_clusters is None:
            self.labels = self.levels.as_clustering().membership
        else:
            self.labels = self.levels.as_clustering(n=self._n_clusters).membership
            
        self.labels = np.array(self.labels)
        
    def _fit_nx():
        pass