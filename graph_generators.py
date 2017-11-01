import numpy as np

def star_clique(n_cliques=3, cliques_size=5, noise=False, random_state=0):
    '''
    write star-clique graph


    parameters:

    number of cliques,
    size of cliques - either list or single number
    (in case of list first number will be considered as a central one)
    noise - proportion of noise edges
    
     TODO
     
     add noise
     change n_cliques, cliques_size 
     check n_cliques == len(cliques_size)
     
    EXAMPLE
    
    adj = star_clique(n_cliques=8, cliques_size=np.array([2,3,5,7,9,3,5,6]))

    '''
    random = np.random.RandomState(random_state)
    
    if isinstance(cliques_size, list) or isinstance(cliques_size, np.ndarray):
        
        n_nodes = np.sum(cliques_size)
        
    elif isinstance(cliques_size, int):
        
        n_nodes = n_cliques * cliques_size
        cliques_size = np.ones(n_cliques, dtype=int) * cliques_size
        
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    cum_sizes = np.cumsum(cliques_size)
    
    central_start = 0
    central_end = cliques_size[0]
    
    for idx, c_size in enumerate(cliques_size):
        
        clique = np.ones((c_size, c_size))
        
        start = cum_sizes[idx] - c_size
        end = cum_sizes[idx]
        
        adj[start:end, :][:, start:end] = clique # add cliques
        
        if idx > 0: # add edges between central clique and other ones
            
            i = random.randint(central_start, central_end)
            j = random.randint(start, end)
            
            adj[i, j] = 1
            adj[j, i] = 1  
    
    return adj


def cycle_clique(n_cliques=3, cliques_size=5, noise=False, random_state=0):
    '''
    write cycle clique graph generators

    parameters:

    number of cliques,
    size of cliques - either list or single number
    noise - proportion of noise edges
    
     TODO
     
     add noise
     change n_cliques, cliques_size 
     check n_cliques == len(cliques_size)

    EXAMPLE:
    
    adj_cycle = cycle_clique(n_cliques=8, cliques_size=np.array([2,3,5,7,9,3,5,6]))
    
    '''
    random = np.random.RandomState(random_state)
    
    if isinstance(cliques_size, list) or isinstance(cliques_size, np.ndarray):
        
        n_nodes = np.sum(cliques_size)
        
    elif isinstance(cliques_size, int):
        
        n_nodes = n_cliques * cliques_size
        cliques_size = np.ones(n_cliques, dtype=int) * cliques_size
        
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    cum_sizes = np.cumsum(cliques_size)
    
    for idx, c_size in enumerate(cliques_size):

        clique = np.ones((c_size, c_size))

        start = cum_sizes[idx] - c_size
        end = cum_sizes[idx]

        adj[start:end, :][:, start:end] = clique # add cliques
        
        i = end-1
        j = end
        
        if idx == n_cliques-1:
            i = 0
            j = -1
        
        
        adj[i, j] = 1 # add edges between cliques
        adj[j, i] = 1  
        
    return adj