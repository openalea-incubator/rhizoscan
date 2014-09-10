"""
Implementation of the data structure used to do rsa estimation
"""
import numpy as _np

_DEBUG =None#debug

class RSA_Builder(object):
    """ Used by optimization to id and merge tree path into RSA axes """
    # Axes
    #   - they are stored in a dictionary
    #   - they are not copied when fork, same objects reference are kept, thus
    #   - modifications apply to all fork
    #   - once initialized (& parent set), they are not supposed to be modified
    #
    # ids (key) in axe dict are:
    #   - tuples
    #   - initialized as `(i,)` with incremental `i`
    #   - id of merged axe1 & axe2 is `axe1_id + axe2_id` => unique id
    #
    
    def __init__(self, graph, model, outgoing, path, primary, segment_order, segment_direction):
        """ Construct a RSA_Builder from the tree covering `path` made on `graph`
        
        outgoing:
            A list-of-set graph of the outgoing neighbor of graph segments 
            
        segment_direction: a boolean array of segment direction w.r.t `graph`
            
        ##todo: finish doc
        """
        global _DEBUG
        _DEBUG = self #debug
        
        self.graph = graph
        self.model = model
        self.min_own_length = 0
        
        if path is not None:
            self.segment_angle = (graph.segment.direction()+(2-segment_direction)*_np.pi)%(2*_np.pi)
            
            self.axes = {}
            self.merges = {}
            self.axe_ids = set()
            
            # sort path w.r.t segment_order
            path_tip = {}
            for i,p in enumerate(path):
                if len(p):
                    path_tip.setdefault(p[-1],[]).append(i)
                    
            new_primary = set()
            sorted_path = []
            for s in segment_order:
                if s in path_tip:
                    for p in path_tip[s]:
                        if p in primary:
                            new_primary.add(len(sorted_path))
                        sorted_path.append(path[p])

            path = sorted_path
            primary = new_primary
            
            # construct initial axes from paths
            for axe_id,segments in enumerate(path):
                if axe_id in primary:
                    primary.remove(axe_id)
                    primary.add((axe_id,))
                axe_id = (axe_id,)
                plant = graph.segment.seed[segments[0]]
                axe = BuilderAxe(builder=self, segments=segments, plant=plant,
                                 out_segments=outgoing[segments[-1]])
                
                self.axes[axe_id] = axe
                self.axe_ids.add(axe_id)
                
            self._compute_axes_per_segment()
            self._set_axe_type(primary)


    def fork(self):
        """ Create a fork of this object: deep copy non-shared attributes only """
        fork = RSA_Builder(graph=None, model=None, 
                           outgoing=None, path=None, primary=None,
                           segment_order=None, segment_direction=None)
        
        # shallow copy
        # ------------
        fork.graph = self.graph
        fork.axes = self.axes
        fork.merges = self.merges
        fork.segment_angle = self.segment_angle

        # deep copy
        # ---------
        fork.model        = self.model.copy()
        fork.axe_ids      = self.axe_ids.copy()
        fork.axe_count    = self.axe_count.copy()
        fork.segment_axes = map(set.copy, self.segment_axes)
        
        fork.min_own_length = self.min_own_length
         
        return fork
       
    def _compute_axes_per_segment(self):
        """ compute the list of axes passing through all segments """
        s_axes = [set() for i in xrange(self.graph.segment.number())]
        
        for axe_id, axe in self.axe_iter():
            for segment in axe.segments:
                s_axes[segment].add(axe_id)
        
        self.segment_axes = s_axes
        self.axe_count = _np.vectorize(len)(s_axes)
       
    def _set_axe_type(self, primary):
        """ Select given axes as 'primary', and the other as 'lateral' """
        
        # set attributes of order 1 axes
        for axe_id in primary:
            axe = self.get_axe(axe_id)
            axe.set_parent(None,0)
            axe.set_type('primary')
        
        # set attributes of order 2 axes
        for axe_id, axe in self.axe_iter():
            if axe_id in primary: continue
            
            # find parent with biggest overlap
            parent = None
            start  = -1
            for parent_id in primary:
                p_axe = self.get_axe(parent_id)
                s = find_split_segment(axe.segments,p_axe.segments)
                if s>start:
                    parent = parent_id
                    start  = s
            
            # set parent & type
            axe.set_parent(parent, start)
            axe.set_type('lateral')
        

    # edition
    # -------
    def remove_axe(self, axe_id):
        """ remove axe with id `axe_id` **in-place** """
        self.axe_ids.remove(axe_id)
        for s in self.axes[axe_id].segments:
            self.segment_axes[s].remove(axe_id)
            self.axe_count[s] -= 1
        
    def add_axe(self, axe_id):
        """ Add axe already created """
        self.axe_ids.add(axe_id)
        for s in self.axes[axe_id].segments:
            self.segment_axes[s].add(axe_id)
            self.axe_count[s] += 1
        
    def prune_axes(self, min_length, terminal=True):
        """ Iteratively remove axe with too little tip 
        
        Axe tips are the set of segments at the end of the axes that are not
        cevered by any other axe.
        
        Return a fork of this builder with removed axes
        """
        builder = self.fork()
        builder.min_own_length = min_length
        
        gsegment = builder.graph.segment
        slength = gsegment.length().copy()
        slength[gsegment.seed>0] = 0
        
        if terminal: 
            prunable = [axe_id for axe_id, axe in builder.axe_iter() if len(axe.out_segments)==0]
            
            def min_length():
                axe_tip = builder.axe_tip(True)
                tip_len = dict((aid,slength[slist].sum()) for aid,slist in axe_tip.iteritems() if aid in prunable)
                axe_min = min(tip_len,key=tip_len.get)
                
                return axe_min, tip_len[axe_min]
            
        else:
            def min_length():
                slen = slength * (builder.axe_count<2)
                own_len = dict((aid,slen[axe.segments].sum()) for aid,axe in builder.axe_iter())
                axe_min = min(own_len,key=own_len.get)
                
                return axe_min, own_len[axe_min]
            
        
        while True:
            axe_id, own_len = min_length()
            if own_len>builder.min_own_length: break
            builder.remove_axe(axe_id)
            
        return builder
              
    # utilities
    # ---------
    def get_axe(self, axe_id):
        """ return BuilderAxe object relative to `axe_id` """
        return self.axes.get(axe_id)
    
    def get_axes_at(self, segment):
        """ return the list of axes passing through `segment` """
        return self.segment_axes[segment]
        
    def axe_iter(self):
        """ return an iterator of (id,axe) over this builder axes """
        return ((aid,self.axes[aid]) for aid in self.axe_ids)
        
    def lateral_axes(self):
        return (self.axes[aid] for aid in self.axe_ids if self.axes[aid].order>1) 
        
    def axe_tip(self, uncovered=False):
        """ return the dict of tip segments of all axes
        
        uncovered
          If False, return the path tip element (as a dict)
          if True, return the tip segments uncovered by other axes (dict of list)
        """
        if uncovered:
            axe_tip = {}
            for axe_id, axe in list(self.axe_iter()):
                segments = axe.segments
                tip_start = self.axe_count[segments]==1
                
                if tip_start.all():
                    axe_tip[axe_id] = segments
                else:
                    tip_start = tip_start.size - _np.argmin(tip_start[::-1])
                    axe_tip[axe_id] = segments[tip_start:]
        else:
            axe_tip = dict((aid,axe.segments[-1]) for aid,axe in self.axe_iter())
            
        return axe_tip


    def min_axe_number(self, segments):
        """ return the minimum number axes passing through any `segments` """
        return min(map(len,(self.get_axes_at(s) for s in segments)))
        
    def state(self):
        return frozenset(self.axe_ids)
    def value(self):
        return self.model.value()
    # merging
    # -------
    def get_merge(self, axe1_id, axe2_id, merge_segment):
        """ Get merge dictionary, creating merged axe (axe1-axe2), if necessary
        
        `merge_segment`should:
          - be in `axe1.out_segments`
          - be part of `axe2.segments`
        
        Creating the merge does not add it to available axe set.
        
        See also: `possible_merge`, `evaluate_merges`, `apply_merge`
        """
        merge_key = (axe1_id,axe2_id)
        if merge_key not in self.merges:
            merge = self._make_merge(axe1_id, axe2_id, merge_segment, merge_key)
            self.merges[merge_key] = merge
            
        return merge_key, self.merges[merge_key]

    def _make_merge(self, axe1_id, axe2_id, merge_segment, merge_key):
        """ private method that actually create a merge dict"""
        axe1 = self.get_axe(axe1_id)
        axe2 = self.get_axe(axe2_id)
        
        # cannot merge order 1 axe
        if axe1.order==1 or axe2.order==1:
            return None
        
        # create merge dict
        # -----------------
        merge_key = (axe1_id,axe2_id)
        new_axe_id = axe1_id+axe2_id
        
        merge_position = axe2.segments.index(merge_segment)
        
        merge = dict(axe1=axe1, axe2=axe2, merged_id=new_axe_id,
                     position=merge_position, segment=merge_segment)

        # create merged axe if it don't exist already
        # -----------------
        if new_axe_id not in self.axes:
            # get segments to be merged from axe2
            axe2_end = axe2.segments[merge_position:]
            
            # create merged axe
            new_axe_segments = axe1.segments + axe2_end
            new_axe = BuilderAxe(builder=self, segments=new_axe_segments,
                                 out_segments=axe2.out_segments, 
                                 type=axe1.get_type())
            new_axe.set_parent(axe1.parent, axe1.branch_index)
            
            self.axes[new_axe_id] = new_axe
            
        merge['merged_axe'] = self.axes[new_axe_id]

        return merge

    def possible_merges(self, constraint=False):
        # get all merges grouped by start axe
        #   merges[axe1_id][out_segment] => axe2_ids
        merges = {}
        
        for axe1_id, axe1 in self.axe_iter():
            for out_segment in axe1.out_segments.keys():
                for axe2_id in self.get_axes_at(out_segment):
                    mkey, merge = self.get_merge(axe1_id,axe2_id,out_segment)
                    if merge is None: continue
                    
                    # keep it if axe2 start can be removed
                    #   i.e. it is covered by at least another axe
                    axe2_start = merge['axe2'].segments
                    axe2_start = axe2_start[:merge['position']]
                    if self.min_axe_number(axe2_start)>1:
                        merges.setdefault(axe1_id,{}).setdefault(out_segment,[]).append(axe2_id)
                        
        if constraint:
            # if:
            #   - there is only 1 merging segment, and
            #   - all axe come from same previous segment
            # then keep only merge on:
            #   - the root on the same side (superposition), and
            #   - roots on opposite side if it split out close enough (crossing)
                
            crossing_length = self.min_own_length ## add a dedicated param?
            
            for axe1_id, merging in merges.iteritems():
                if len(merging)>1: continue
                
                axe1 = self.get_axe(axe1_id)
                
                merge_seg  = merging.keys()[0]
                merge_axe_ids = merging[merge_seg]
                merge_axes = map(self.axes.get,merge_axe_ids)
                merge_curves = (axe.get_curve_from(merge_seg) for axe in merge_axes)
                merge_curves = dict(zip(merge_axe_ids,merge_curves))
                
                origin = [c[0][0] for c in merge_curves.values()]
                merge_axe_angle = origin[0]
                if origin.count(merge_axe_angle)!=len(origin):  # if not all equal
                    continue
                
                # axe selection for 'superposition' side
                side = axe1.out_segments[merge_seg]>merge_axe_angle
                select = max if side else min
                
                # select superposition
                keeped = [select(merge_curves,key=merge_curves.get)]
                
                if crossing_length:
                    # sort curves
                    curves = sorted(((c,aid) for aid,c in merge_curves.iteritems()), reverse=side)
                    # separate curves and their merge_axe_ids
                    curves, maxe_ids = zip(*curves)
                    # zip curves s.t. curves[i]=all curves ith segment
                    #   add fake if for shorter curves
                    max_len = max(map(len,curves))-1
                    fake    = ((0,crossing_length+1),)
                    curves = zip(*[c+(max_len-len(c))*fake for c in curves])
                    
                    cur_pos  = 0
                    cur_dist = 0
                    cur_split = 0
                    
                    while True:                
                        c = curves[cur_pos]
                        if cur_split: c = c[:-cur_split]
                        
                        c_ref = c[0]
                        cur_split += sum([ci!=c_ref for ci in c])
                        
                        cur_pos += 1
                        cur_dist += c_ref[1]
                        
                        if cur_pos>=len(curves) or cur_dist>crossing_length:
                            break
                        
                    if cur_split:  # 
                        keeped = maxe_ids[-cur_split:]
                                         
                # udpate merging dict                                           
                merging[merge_seg] = keeped                                           
 
                        
        # return "flattened" merges
        # -------------------------
        #   output[(axe1_id,axe2_id)] = 
        #      dict(segment=..., position=..., axe_id=...)
        output = {}  
        for axe1_id, merging in merges.iteritems():
            for merge_segment, axe2_ids in merging.iteritems(): 
                for axe2_id in axe2_ids:
                    key, merge = self.get_merge(axe1_id,axe2_id,merge_segment)
                    merge = merge.copy()
                    
                    # set merged state
                    state = self.axe_ids.copy()
                    state.remove(axe1_id)
                    state.remove(axe2_id)
                    state.add(merge['merged_id'])
                    merge['state'] = frozenset(state)
            
                    output[key] = merge
                        
        return output
            
    def evaluate_merge(self, merge):
        """ Evaluate model value of given `merge`
        
        `merge` should be one of value of the dict returned by `possible_merges`
        
        Return the evaluated model value.
        The `merge` input is updated in order to be usable by `apply_merge`
        """
        axe1 = merge['axe1']
        axe2 = merge['axe2']
        axeM = merge['merged_axe']
        
        value, new_param = self.model.evaluate_merge(self, axe1=axe1, axe2=axe2, merged=axeM)
        
        #Dprint id(self.model), id(self.model.param), self.model.param[4],\
        #D  id(new_param), new_param[4],\
        #D  self.model.distribution.std(new_param)
        #D  #debug
        #Dmerge['model'] = self.model
        #Dmerge['mparam'] = self.model.param
              
        merge['param'] = new_param  #! in-place !
        merge['value'] = value      #

        return value
        
    def apply_merge(self, merge_key, merge, update=False):
        """ apply a merging obtained from `evaluate_merge()` """
        up_param = merge['param']
        #Dprint ' hist:', self.model.history
        #Dprint ' merge:', merge_key
        #Dprint ' debug:', id(self.model), id(self.model.param), self.model.param[4], id(up_param), merge['param'][4]
        #D_model = merge['model']
        #Dif _model is not self.model:
        #D    None+1
        
        fork = self.fork()
        axe1_id, axe2_id = merge_key
        
        # update axes
        # -----------
        fork.remove_axe(axe1_id)
        fork.remove_axe(axe2_id)
        fork.add_axe(merge['merged_id'])

        # update model (if required)
        # ------------
        if update:
            fork.model.update(modification=merge_key, parameters=merge['param'])
            
        ##fork.merges.pop(merge_key)

        return fork

    # make RootTree
    def make_tree(self, prune='min_own_length'):
        """ Make a RootTree from this builder """
        from rhizoscan.root.graph import AxeList
        from rhizoscan.root.graph import RootTree
        
        #fork = self.fork()
        if prune=='min_own_length':
            prune = self.min_own_length
        
        fork = self.prune_axes(min_length=prune, terminal=False)
        
        int_id = {}
        for ind, (axe_id,axe) in enumerate(fork.axe_iter()):
            int_id[axe_id] = ind+1

        axe_num = len(fork.axe_ids)+1
        
        segments = [[] for i in range(axe_num)]
        parent   = _np.zeros(axe_num, dtype=int)
        sparent  = _np.zeros(axe_num, dtype=int)
        plant    = _np.zeros(axe_num, dtype=int)
        order    = _np.zeros(axe_num, dtype='uint8')
        ids      = _np.zeros(axe_num, dtype=int)
        bids     = _np.zeros(axe_num, dtype=int)##[None]*axe_num   #debug
        
        for i, (axe_id, axe) in enumerate(fork.axe_iter()):
            i += 1
            branch_ind  = axe.branch_index
            segments[i] = axe.segments[branch_ind:]
            parent[i]   = int_id[axe.parent] if axe.parent is not None else 0
            sparent[i]  = axe.segments[branch_ind-1] if branch_ind else 0
            plant[i]    = axe.plant
            order[i]    = axe.order
            ids[i]      = int_id[axe_id]
            bids[i]     = axe_id[0]  #debug
        
        graph = fork.graph
        axe = AxeList(axes=segments, segment_list=graph.segment, 
                      parent=parent, parent_segment=sparent,
                      plant=plant,   order=order,    ids=ids)
        axe.builder_id = bids  #debug
        
        return RootTree(node=graph.node,segment=graph.segment, axe=axe)


class BuilderAxe(object):
    """ Store one axe for a RSA_Builder """
    def __init__(self, builder, segments, out_segments, type=None, plant=None):

        def angle_diff(angle1, angle2):  #angle1:next, angle2:prev
            dangle = angle1-angle2
            return min(dangle, 2*_np.pi-dangle)

        gsegments     = builder.graph.segment
        segment_angle = builder.segment_angle


        self.builder  = builder
        self.segments = segments
        self.plant = plant

        # compute axe curve: list of (angle-derivative,length)
        seg_len = gsegments.length()
        seg_len = seg_len[self.segments]
        
        seg_angle = segment_angle[segments].tolist()
        seg_angle = map(angle_diff, seg_angle, [seg_angle[0]] + seg_angle[:-1])

        self.curve = tuple(zip(seg_angle, seg_len))
        
        # set outgoing segments
        term_angle = segment_angle[segments[-1]]
        out = dict((s,angle_diff(segment_angle[s], term_angle)) for s in out_segments) 
        self.out_segments = out
        
        # compute axe length
        self.length = seg_len.sum()
        
        # set type
        self.set_type(type)
        
        
    # architecture properties
    # -----------------------
    def set_type(self, type):
        """ Set axe type """
        self._type = None if type is None else str(type).lower()
        
    def get_type(self):
        return self._type
        
    def get_curve_from(self, segment):
        """ return axe curve starting at `segment` """
        return self.curve[self.segments.index(segment):]
        
    def set_parent(self, parent, branch_index):
        """ set parent branch of axes
        
        `parent`
            id of the parent root axe
        `branch_index`
            index of the first segment after branching
            
        Note: parent axe should have its own parent already set
        """
        self.parent = parent
        self.branch_index = branch_index
        
        if parent is None:
            self.order = 1
            self.branch_length = self.length
            
        else:
            parent = self.builder.get_axe(parent)
            self.plant = parent.plant
            self.order = parent.order+1
            
            # compute branch length
            seg_len = self.builder.graph.segment.length()
            self.branch_length = seg_len[self.segments[branch_index:]].sum() 


# utils
# -----
def find_split_segment(path1,path2):
    """ return index of 1st element of `path1` that is not in `path2` """
    path_cmp = map(lambda x,y: x!=y, path1,path2)
    if True in path_cmp:
        return path_cmp.index(True)
    else:
        return 0

