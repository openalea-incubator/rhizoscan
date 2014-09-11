"""
This package contains functionalities to estimate a Root System Architecture
embeded in a (2D) graph.
"""


def estimate_RSA(graph, model='arabidopsis', min_length=10, init_axes=None, verbose=1, **optim_kargs):
    """
    Find a best fit RSA embeded in `graph`
    
    :Inputs:
      graph
        A `RootGraph` object into which the rsa is embeded
      model
        The RSA_Model used for optimisation - see the `model` module
      min_length
        Minimum length under which measurements are not safe:
          - Root axes of lower length are not processed
          - branching under min_length are/can be a crossing ##?
      init_axes
        Optional `AxeList` object, constructed on `graph`, to used as basis
        
    :Output:
      A `Treegraph` object representing the estimated RSA
    """
    import numpy as _np
    from rhizoscan.root.rsa.dag import graph_to_dag
    from rhizoscan.root.rsa.dag import least_curvature_tree
    from rhizoscan.root.rsa.dag import minimum_dag_branching
    from rhizoscan.root.rsa.dag import dag_topsort
    from rhizoscan.root.rsa.dag import tree_covering_path
    
    from rhizoscan.root.rsa.builder import RSA_Builder
    
    # initialize variables
    # --------------------
    segment = graph.segment
    length = segment.length()
    angle = segment.direction_difference()
    src  = (graph.segment.seed>0) 
    
    if isinstance(model,basestring):
        from rhizoscan.root.rsa.model import get_model
        model = get_model(model)
    
    from rhizoscan.root.rsa.optimizer import parse_all
    optim = parse_all


    # graph to DAG
    # ------------
    dag, sdir = graph_to_dag(segment, init_axes=init_axes)
    dag_in, dag_out = zip(*dag)
    top_order = dag_topsort(dag=dag, source=src)
    
    # select primary axes
    # -------------------
    #   find path which minimize cumulative curvature from seed
    parent,curv = least_curvature_tree(dag_out,src,angle,length, init_axes=init_axes)
    path_elt,elt_path,init_map = tree_covering_path(parent=parent, top_order=top_order, init_axes=init_axes)
    
    #   select "best" primary ##TODO: move that somewhere else...
    #     1. possible primary axes per plant
    #     2. select longest
    #     3. update/create init_axes with select primary
    primary = set()  # path indices of selected primary
    
    def path_len(path_ind):
        return path_ind, length[path_elt[path_ind]].sum()

    if init_axes:
        path_plant = {}
        init_primary = (init_axes.order()==1).nonzero()
        
        for axe_ind in init_primary:  # 1 primary per plant expected
            plant = init_axes.plant[axe_ind]
            plant_path[plant] = dict(map(path_len, init_map[axe_ind]))
            
            selected = max(path_length, key=path_length.get)
            primary.add(selected)
            
            # update init_axes
            init_axes.segment[axe_ind] = path_elt[selected]
            
        init_axes.clear_temporary_attribute()
        
    else:
        plant_path = {}
        for path_ind,elt_list in enumerate(path_elt):
            if len(elt_list)==0: continue
            plant = segment.seed[elt_list[0]]
            plant_path.setdefault(plant,[]).append(path_ind) 
            
        for plant, path_indices in plant_path.iteritems():
            plant_path[plant] = dict(map(path_len, path_indices))
            
        # select primary
        for plant,path_length in plant_path.iteritems(): 
            primary.add(max(path_length, key=path_length.get))
        
        # create init_axes with selected primary        
        from rhizoscan.root.graph import AxeList
        zeros = _np.zeros(len(primary),dtype=int)
        ones  = _np.ones(len(primary),dtype=int)
        init_axes = AxeList(axes=[path_elt[s] for s in primary],
                            segment_list=segment,
                            parent=zeros,
                            order=ones,
                            parent_segment=zeros.copy())
    
    # find secondary axes
    # -------------------
    # tree convering path 
    parent = minimum_dag_branching(incomming=dag_in, cost=angle, init_axes=init_axes)
    path_elt,elt_path,init_map = tree_covering_path(parent=parent, top_order=top_order, init_axes=init_axes)
    
    
    # construct builder
    builder = RSA_Builder(graph, path=path_elt, primary=primary, 
                          segment_order=top_order, segment_direction=sdir, 
                          outgoing=dag_out, model=model)
    builder = builder.prune_axes(min_length=min_length)
    
    # optimization of lateral root merging
    if len(tuple(builder.lateral_axes()))>1:
        builder.model.fit(builder, [axe for aid,axe in builder.lateral_axes()])
        builder = optim(builder=builder, update_model=True, verbose=verbose, **optim_kargs)
    
    
    # build and return TreeGraph
    # --------------------------
    return builder.make_tree()
    
    