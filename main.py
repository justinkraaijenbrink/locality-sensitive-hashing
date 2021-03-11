import numpy as np
import pandas as pd
import scipy.sparse as sps
import itertools
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", help = "Data file path", type = str)
parser.add_argument("-s", help = "Random seed", type = int, default = 2020)
parser.add_argument("-m", help = " Similarity measure (js / cs / dcs)", type = str)
args = parser.parse_args()

def LoadData(path_to_data):
    filepath = path_to_data
    data = np.load(filepath)
    return(data)

def JaccardSimilarity(S1, S2):
    
    num = float(S1.multiply(S2).sum())
    denom = float((S1 + S2).sum())
    
    S = float(num / denom)
    
    return(S)

def SignSimilarity(sign_S1, sign_S2):
    
    S = float(len(np.where(sign_S1==sign_S2)[0]))/float(len(sign_S1))
    
    return(float(S))

def CreateSparseMatrix(data):
    users = data[:, 0]
    movies = data[:, 1]
    n_users = np.max(users) + 1
    n_movies = np.max(movies) + 1
    
    rated = np.ones(len(data), dtype = 'bool')
    
    smc = sps.csc_matrix((rated, (movies, users)), shape = (n_movies, n_users), dtype = 'bool')
    smr = sps.csr_matrix(smc)
    smco = sps.coo_matrix(smr)
    smlil = sps.lil_matrix(smco)
    out = {'smc' : smc, 
           'smr' : smr, 
           'smco' : smco,
           'smlil' : smlil}
    
    return(out)

def Minhash(sparsemat, num_iter = 100):
    num_movies, num_users = np.shape(sparsemat)
    signatures = np.zeros([num_iter, num_users])
    
    for i in range(num_iter):
        np.random.seed(i)
        index = np.random.permutation(num_movies)
        permutedmatrix = sparsemat[index, :]

        for j in range(1, num_users):
            hashval = permutedmatrix.indices[permutedmatrix.indptr[j]:permutedmatrix.indptr[j+1]].min()
            signatures[i, j] = hashval
            
    return(signatures)

def PerformLSH(signatures, rows = 6):
    candidates = set()
    
    n_bands = int(np.floor(signatures.shape[0] / rows))

    for r in range(n_bands):
        print('bands:', r, 'out of', n_bands-1)
        band = signatures[r * rows:(r + 1) * rows, :]

        comb, index = np.unique(band, axis = 1, return_inverse = True)

        s_id = np.argsort(index)
        sorted_index = index[s_id]

        buckets = np.array(np.split(s_id, np.nonzero(sorted_index[1:] > sorted_index[:-1])[0] + 1))
        
        for b in buckets:
            if len(b) == 2:
                pair = tuple(np.sort([b[0], b[1]]))
                candidates.add(pair)
                    
            if len(b) > 2:
                combinations = list(map(list, itertools.combinations(b, 2)))
                for c in combinations:
                    pair = tuple(np.sort([c[0], c[1]]))
                    candidates.add(pair)
                    
    return(candidates)

def FindPairs(candidates, sparsematrix, signmatrix, signthresh = 0.4, jsthresh = 0.5):
    similar_pairs = []
    
    for c in enumerate(candidates):
        sim = SignSimilarity(signmatrix[:,c[1][0]], signmatrix[:,c[1][1]])
        if sim > signthresh:
            JS = JaccardSimilarity(sparsematrix[:, c[1][0]], sparsematrix[:, c[1][1]])
            if JS > jsthresh:
                similar_pairs.append(c[1])
                np.savetxt('js.txt', sorted(similar_pairs), delimiter=',', fmt='%i')
    print('Number of pairs with similarity above treshold:', len(similar_pairs))
    return(similar_pairs)

def CosineSimilarity(S1, S2):
    num = float(S1.multiply(S2).sum())
    denom = float(np.sqrt(S1.power(2).sum())) * float(np.sqrt(S2.power(2).sum()))
    
    theta = np.arccos(num / denom)
    
    sim = 1 - theta / np.pi
    return(sim)

def CreateSparseMatrixCos(data, discrete = False):
    dtype = "int8"
    if discrete == True:
        dtype = "bool"
    
    users = data[:, 0]
    movies = data[:, 1]
    n_users = np.max(users) + 1
    n_movies = np.max(movies) + 1
    
    ratings = data[:, 2]
    
    smc = sps.csc_matrix((ratings, (movies, users)), shape = (n_movies, n_users), dtype = dtype)
    smr = sps.csr_matrix(smc)
    smco = sps.coo_matrix(smr)
    smlil = sps.lil_matrix(smco)
    out = {'smc' : smc, 
           'smr' : smr, 
           'smco' : smco,
           'smlil' : smlil}
    
    return(out)

def CreateSketch(u, num_vectors = 150):
    num_movies = u.shape[0]
    v = np.random.choice([-1, 1], num_movies*num_vectors).reshape([num_vectors, num_movies])
    
    num_users = u.shape[1]
    sketch = v * u
    sketch[sketch >= 0] = 1
    sketch[sketch < 0] = -1
    
    return sketch

def FindCandidatesCos(signatures, rows = 20):
    candidates = set()

    n_bands = int(np.floor(signatures.shape[0] / rows))

    for r in range(n_bands):
        print('band:', r, "out of", n_bands-1)
        band = signatures[r * rows:(r + 1) * rows, :]

        comb, index = np.unique(band, axis = 1, return_inverse = True)

        s_id = np.argsort(index)
        sorted_index = index[s_id]

        buckets = np.array(np.split(s_id, np.nonzero(sorted_index[1:] > sorted_index[:-1])[0] + 1))
        
        for b in buckets:
            if len(b) == 2:            
                pair = tuple(np.sort([b[0], b[1]]))
                candidates.add(pair)
                
            if len(b) > 2:
                combinations = list(map(list, itertools.combinations(b, 2)))
                for c in combinations:
                    pair = tuple(np.sort([c[0], c[1]]))
                    candidates.add(pair)
                    
    return(candidates)

def FindSimilarPairsCos(candidates, signatures, sparsematrix, thresh = 0.73, cos_thresh = 0.73):
    sim_pairs = []

    for p in candidates:
        s1 = signatures[:, p[0]]
        s2 = signatures[:, p[1]]
        sim = float(len(np.where(s1 == s2)[0]))/float(len(s1))

        if sim > thresh:
            ## Extra validation step, described in report
            #sim_cos = CosineSimilarity(sparsematrix[:, p[0]], sparsematrix[:, p[1]])
            #if sim_cos > cos_thresh:
            sim_pairs.append(p)
            np.savetxt('cs.txt', sorted(sim_pairs), delimiter=',', fmt='%i')
    print('Number of pairs with similarity above treshold:', len(sim_pairs))
    return(sim_pairs)

def FindSimilarPairsDCos(candidates, signatures, sparsematrix, thresh = 0.73, cos_thresh = 0.73):
    sim_pairs = []

    for p in candidates:
        s1 = signatures[:, p[0]]
        s2 = signatures[:, p[1]]
        sim = float(len(np.where(s1 == s2)[0]))/float(len(s1))

        if sim > thresh:
            ## Extra validation steps, described in report
            #sim_cos = CosineSimilarity(sparsematrix[:, p[0]], sparsematrix[:, p[1]])
            #if sim_cos > cos_thresh:
            sim_pairs.append(p)
            np.savetxt('dcs.txt', sorted(sim_pairs), delimiter=',', fmt='%i')
    print('Number of pairs with similarity above treshold:', len(sim_pairs))
    return(sim_pairs)


if __name__ == '__main__':
    time_start = time.time()
    np.random.seed(seed=args.s)
    print("Loading data...")
    data = LoadData(args.d)
    
    if (args.m == 'js'):
        print("Creating sparse matrix...")
        sparsemat = CreateSparseMatrix(data)
        u = sparsemat["smc"]
        print("Creating signature matrix...")
        signatures = Minhash(u, num_iter = 150)
        print("Creating candidate set...")
        candidates = PerformLSH(signatures, rows = 6)
        print("Finding pairs with similarity above treshold...")
        sim_pairs = FindPairs(candidates, u, signatures, signthresh = 0.42, jsthresh = 0.5)

    if (args.m == 'cs'):
        print("Creating sparse matrix...")
        smc_cos = CreateSparseMatrixCos(data, discrete = False)
        u_cos = smc_cos['smc']
        print("Creating sketches...")
        signatures = CreateSketch(u_cos, num_vectors = 90)
        print("Creating candidate set...")
        candidates = FindCandidatesCos(signatures, rows = 30)
        print("Finding pairs with similarity above treshold...")
        sim_pairs = FindSimilarPairsCos(candidates, signatures, u_cos, thresh = 0.73)
        
    if (args.m == 'dcs'):
        print("Creating sparse matrix...")
        smc_cos = CreateSparseMatrixCos(data, discrete = True)
        u_cos = smc_cos['smc']
        print("Creating sketches...")
        signatures = CreateSketch(u_cos, num_vectors = 90)
        print("Creating candidate set...")
        candidates = FindCandidatesCos(signatures, rows = 30)
        print("Finding pairs with similarity above treshold...")
        sim_pairs = FindSimilarPairsDCos(candidates, signatures, u_cos, thresh = 0.73)
              
    print('Total time elapsed:', (time.time() - time_start)//60, 'mins and', round((time.time() - time_start)%60, 1), 'seconds.')
