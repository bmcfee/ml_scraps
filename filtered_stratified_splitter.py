import numpy as np

def filtered_stratified_split(ids, splitter, Y, **kwargs):
    '''Cross-validation split filtration. Ensures that points of the same meta-id 
    end up on the same side of the split

    input:

        ids:         n-by-1 mapping of data points to meta-id
        splitter:    handle to the cross-validation class (eg, StratifiedShuffleSplit)
        Y:           n-by-1 vector of class labels
        **kwargs:    arguments to the cross-validation class

    yields:
        (train, test) indices
    '''
    
    n = len(Y)
    
    indices = ('indices' in kwargs) and (kwargs['indices'])   
    
    kwargs['indices'] = True
    
    def unfold(meta_ids, X_id, indices):
        split_ids = []
        for i in meta_ids:
            split_ids.extend(X_id[i])
            
        split_ids = np.array(split_ids)
        
        if not indices:
            z = np.zeros(n, dtype=bool)
            z[split_ids] = True
            return z
    
    # 1: make a new label vector Yid
    X_id = []
    Y_id = []
    
    last_id = None
    for i in xrange(len(ids)):
        if i > 0 and last_id == ids[i]:
            X_id[-1].append(i)
        else:
            last_id = ids[i]
            X_id.append([i])
            Y_id.append(Y[i])
            
    # 2: CV split on Yid
    splits = splitter(Y_id, **kwargs)
    
    # 3: Map CV indices back to Y space
    for meta_train, meta_test in splits:
        yield (unfold(meta_train, X_id, indices), 
               unfold(meta_test, X_id, indices))
