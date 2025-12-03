import numpy as np
from numpy.fft import fft, ifft
# generate random normalized vectors for roles/concepts
def rand_vec(D): return np.random.normal(size=D)
def normalize(v): return v/np.linalg.norm(v)
def bind(r,f):                       # HRR bind via FFT
    return np.real(ifft(fft(r)*fft(f)))
def unbind(r,c):                     # approximate unbind
    return np.real(ifft(fft(c)*np.conj(fft(r))))
# cleanup memory via nearest neighbor in semantic dict
def cleanup(v, store):
    # store: dict{name:vector}
    names = list(store.keys()); vecs = np.vstack(list(store.values()))
    sims = vecs @ v / (np.linalg.norm(vecs,axis=1)*np.linalg.norm(v))
    return names[np.argmax(sims)]
# small ontology: vectors + adjacency
D = 512
semantic_store = {'apple': normalize(rand_vec(D)), 'box': normalize(rand_vec(D))}
adj = {}  # edge list
# perception: seen object 'apple' in 'box' role
r = normalize(rand_vec(D))           # role vector e.g. LOCATION
f = semantic_store['apple']
c = bind(r,f)
# retrieval test
f_hat = unbind(r,c)
label = cleanup(f_hat, semantic_store)  # expected 'apple'
# ontology expansion: new concept from fused exemplars
new_vec = normalize(0.6*semantic_store['apple'] + 0.4*normalize(rand_vec(D)))
semantic_store['apple_variant'] = new_vec
adj[('apple','variant_of')] = 'apple_variant'