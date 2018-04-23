"""
Implements model to be optimized during RSA estimation
"""

def get_model(name):
    if name.lower()=='arabidopsis':
        return Model(evaluators=[ConstantGrowthEvaluator(), 
                                 MinBranchingSuperpositionEvaluator()],
                     coefficients=[1,1])
    else:
        raise TypeError('unrecognized model with name ' + str(name))

class Model(object):
    def __init__(self, evaluators, coefficients, history=[]):
        self.evaluators   = evaluators
        self.coefficients = coefficients
        self.history = history
    
    def copy(self):
        from copy import copy
        cp = self.__class__(evaluators=[e.copy() for e in self.evaluators],
                            coefficients=copy(self.coefficients),
                            history=copy(self.history))
        return cp
        
    def fit(self, builder, axes):
        axes = tuple(axes)
        for e in self.evaluators:
            e.fit(builder=builder, axes=axes)
            
    def value(self, params=None):
        """ return Model value from param, or model evaluator param """
        v = 0
        c = self.coefficients
        for i,e in enumerate(self.evaluators):
            v += c[i]*e.value(param=params[i] if params else None)
            
        return v
            
    def evaluate_merge(self, builder, axe1, axe2, merged):
        """ Evaluate Model evaluators if merging axe1 and axe2 into merged 
        
        Return model value and evaluators parameter
        """
        up_params = []
        for e in self.evaluators:
            up_params.append(e.evaluate_merge(builder, axe1, axe2, merged))
            
        return self.value(up_params), up_params

    def update(self, modification, parameters):
        """ Update evaluators with given `parameters` and record it """
        self.history.append(modification)
        for e,p in zip(self.evaluators,parameters):
            e.set_parameters(p)
        

class AbstractEvaluator(object):
    """ Abstract base class for evaluators """
    def __init__(self, parameters=[]):
        self.param=parameters
        
    def copy(self):
        from copy import deepcopy as dc
        return self.__class__(parameters=dc(self.param))

    def set_parameters(self, parameters):
        self.param = parameters

    # automated axe measurement
    # -------------------------
    #   used only for measure that are only dependent on the axe
    @staticmethod
    def measure(axe, builder):
        raise NotImplementedError('Abstract class method')
        
    def axe_measurement(self, axe, builder):
        if not axe.weight: return None
        
        cls_name = self.__class__.__name__
        value = getattr(axe, cls_name,None)
        if value is None:
            value = self.measure(axe,builder)
            setattr(axe,cls_name, value)
        return value

    def axes_measurement(self, axes, builder):
        """ return value of all `axes` using axe_measurement """
        return map(self.axe_measurement, axes, [builder]*len(axes))


    # pure astract method
    # -------------------
    def fit(self, builder, axes):
        """ set Evaluator param from `axes` in `builder` """
        raise NotImplementedError('Abstract class method')
        
    def value(self, param=None):
        """ get Evalutor value for given param or self param if not given """
        raise NotImplementedError('Abstract class method')
        
    def evaluate_merge(self, builder, axe1, axe2, merged):
        """ Evaluate Evaluator param if merging axe1 and axe2 into merged """
        raise NotImplementedError('Abstract class method')
    
class ConstantGrowthEvaluator(AbstractEvaluator):
    """ Architecture model expected from a constant growth hypothesis

    The hypothesis are:
      - All root axes have constant growth
      - Laterals are initiated at a constant distance/delay of the primary tip 
      
    The consequence is that the following ratio should be constant:
    
        lateral-length / ramification-to-primary-tip-length
        
    This evaluator value is the standard deviation of this ratio on all axes
    """
    """
    Note:
      Even with constant growth, lateral speed are different from primary
      thus the ratio is proportional to branching position
      
      This model is not quite good meaningful until the end
      => should add some evaluator that influence iteration locally
      
      what about weighting axes?
       - w.r.t own/tip_length? 
       - if depends on current axe set, then it should be updated for all at
         each step&evaluation...
    """
    from .distributions import NormalDistribution as distribution
    
    # axe length ratio
    # ----------------
    @staticmethod
    def measure(axe, builder):
        axe_len = axe.branch_length
        overlap = axe.length - axe_len
        p_length = builder.get_axe(axe.parent).length
        tip_dist = p_length - overlap
        
        return (tip_dist-axe_len)/p_length
        
    # model API
    # ---------
    def fit(self, builder, axes):
        """ Fit model on `builder` and return model distribution object """
        samples = self.axes_measurement(axes,builder)
        self.param = self.distribution.fit(filter(None,samples))
    
    def value(self, param=None):
        if param is None: param = self.param
        return self.distribution.std(param)
        
    def evaluate_merge(self, builder, axe1, axe2, merged):
        """ evaluate std diff of merging `axe1` & `axe2` by `merged`
        
        Return the updated parameter set to be updated if merge is done  
        """
        from copy import deepcopy
        
        dist = self.distribution
        values = self.axes_measurement([axe1,axe2,merged], builder)
        up_param = deepcopy(self.param)
        
        f = lambda x: filter(None,x)
        dist.replace(up_param, remove=f(values[:2]), add=f([values[2]]))
        
        return up_param
    

#--------------------------
class MinBranchingSuperpositionEvaluator(AbstractEvaluator):
    """ Evaluator that penalizes having several branching on same segment
    
    The cost is the sum of multiple branching on same segment:
      value = sum( branching_number-1 for all branching segment )
    """
    """
    Note:
      Quite simplistic evaluator but has finally good effect:
        - reduce total number of axes
        - reduce number of axe branching at same place
      
      Todo?
        - count instead overlapping axes at branching segment
        - take into account superpostion length, ... or not
        - look at distance between (closest) branching
    """
    @staticmethod
    def measure(axe, builder):
        """ find axe branch segment """
        if axe.weight==0: return None
        b_ind = axe.branch_index
        return None if b_ind is None else axe.segments[b_ind] 
        
    def fit(self, builder, axes):
        from collections import Counter
        branch = Counter(filter(None,self.axes_measurement(axes,builder)))
        self.param = self._make_param(branch)

    @staticmethod
    def _make_param(branch):
        value = sum(max(count-1,0) for count in branch.viewvalues())
        return (branch,value)

    def value(self, param=None):
        if param is None: param = self.param
        return param[1]

    def evaluate_merge(self, builder, axe1, axe2, merged):
        ## DEBUG: check that fitted modelis same as stored ones
        ##ev = MinBranchingSuperpositionEvaluator()
        ##ev.fit(builder, [axe for aid,axe in builder.lateral_axes()])
        ##c1 = builder.model.evaluators[1].param[0]
        ##c2 = ev.param[0]
        ##if not all(c1[k]==c2[k] for k in set(c1.keys()+c2.keys())):
        ##    print ' ***ERROR***' ##
        ##    import pdb
        ##    pdb.set_trace()
        dbranch = self.axes_measurement([axe1,axe2,merged],builder)
        branch = self.param[0].copy()
        f = lambda x: filter(None,x)
        branch.subtract(f(dbranch[:2]))
        if dbranch[2] is not None:
            branch[dbranch[2]] += 1
        return self._make_param(branch)
    
