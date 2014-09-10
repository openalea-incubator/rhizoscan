"""
Implements model to be optimized during RSA estimation
"""

def get_model(name):
    if name.lower()=='arabidopsis':
        return ArabidopsisModel()
    else:
        raise TypeError('unrecognized model with name ' + str(name))

class ArabidopsisModel(object):
    """ Simple architecture model for arabidopsis
    
    Axe type:
      - one 'primary' root per plants - the longest
      - all others are 'lateral'
     
    Architecture model (for lateral merging optimization):
      - model lateral-length/branching-to-main-axe-tip as a normal districution
      - thus, merge probability is maximized when close to this ratio
      
    """
    from .distributions import NormalDistribution as distribution
    
    measure_name = '_AM'
    
    def __init__(self, parameters=[], history=[]):
        self.param   = parameters
        self.history = history
        
    def copy(self):
        from copy import copy as c, deepcopy as dc
        return self.__class__(parameters=dc(self.param), history=c(self.history))
            
    # get and compute axe value
    # -------------------------
    @staticmethod
    def measure(axe, parent):
        axe_len = axe.branch_length
        overlap = axe.length - axe_len
        tip_dist = parent.length - overlap
        
        return (tip_dist-axe_len)/parent.length
        
    def axe_measurement(self, axe, builder):
        value = getattr(axe, self.measure_name,None)
        if value is None:
            value = self.measure(axe,builder.get_axe(axe.parent))
            setattr(axe,self.measure_name, value)
        return value

    def axes_measurement(self, axes, builder):
        """ return value of all `axes` using axe_measurement """
        return map(self.axe_measurement, axes, [builder]*len(axes))

    # fit model parameter
    # -------------------
    def fit(self, builder, axes):
        """ Fit model on `builder` and return model distribution object """
        samples = []
        for axe in axes:
            samples.append(self.axe_measurement(axe,builder))
                                       
        self.param = self.distribution.fit(samples)
    
    def value(self, param=None):
        if param is None: param = self.param
        return param[2]
        return self.distribution.std(param)
        
    # merges
    # ------
    def evaluate_merge(self, builder, axe1, axe2, merged):
        """ evaluate std diff of merging `axe1` & `axe2` by `merged`
        
        :Outputs:
          - difference of standard deviation if merge is done
          - the updated parameter set to be updated if merge is done  
        """
        from copy import deepcopy
        
        dist = self.distribution
        values = self.axes_measurement([axe1,axe2,merged], builder)
        up_param = deepcopy(self.param)
        
        dist.replace(up_param, remove=values[:2], add=[values[2]])
        
        return self.value(up_param), up_param
    
    def update(self, modification, parameters):
        #Ds = self.distribution.std                     #debug
        #Dp = self.distribution.fit(parameters[3])      #debug
        #D
        #Dsp_cur = s(parameters)
        #Dsp_ref = s(p)
        #Dsp_err = abs(sp_cur-sp_ref)/sp_ref                          #debug
        #Dif sp_err>0.0001:                                           #debug
        #D    print '  >ERR updated!=fitted: %f-%f'%(sp_cur,sp_ref)   #debug
            
        self.history.append(modification)
        self.param = parameters

    # axe order and parenting
    # -----------------------
    @staticmethod
    def set_axe_type(builder):
        """ Select axe type, order and set parent of axes in `builder` """
        from .utils import longest_axe
        from .utils import find_split_segment
        
        # select 1st order axes, as the longuest axe per plant
        o1_axes = ((aid,getattr(axe,'order',None)) for aid,axe in builder.axe_iter())
        o1_axes = [aid for aid,order in o1_axes if order==1] 
        o1_axes = longest_axe(axes=builder.axe_iter(), group='plant', selected=o1_axes)

        # dict of (plant, primary-axe-id)
        #   if multiple primary per plant, select the longest
        primaries =       ((builder.get_axe(aid),aid) for aid in o1_axes)
        primaries = sorted((axe.plant,axe.length,aid) for axe,aid in primaries)
        primaries =   dict((plant,aid)                for plant,l,aid in primaries)


        # set attributes of order 1 axes
        for axe_id in o1_axes:
            axe = builder.get_axe(axe_id)
            axe.set_parent(None,0)
            axe.set_type('primary')
        
        
        # set attributes of order 2 axes
        for axe_id, axe in builder.axe_iter():
            if axe_id in o1_axes: continue
            
            # find parent
            #   by default, same plant primary axe
            parent = primaries[axe.plant]
            start  = 0
            
            #   find parent with biggest overlap
            for parent_id in o1_axes:
                p_axe = builder.get_axe(parent_id)
                s = find_split_segment(axe.segments,p_axe.segments)
                if s>start:
                    parent = parent_id
                    start  = s
            
            # set parent & type
            axe.set_parent(parent, start)
            axe.set_type('lateral')
        

#--------------------------
class GeometricalModel(object):
    """ Probabilistic model that "push" toward logical geometrical branching """
    def __init__(self, crossing_length):
        """ Make a GeometricalModel 
        
        `crossing_length` 
            length under which root superposition are expected to be crossings
        """
        self.crossing_length = crossing_length
        
    def fit(self, builder):
        pass
    
    def evaluate(self, builder):
        pass
    
    def merge(self, builder, axes):
        pass
    
