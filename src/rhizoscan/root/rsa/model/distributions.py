"""
Implement normal distribution
"""
import math

class NormalDistribution(object):
    @classmethod
    def fit(self, values):
        """ fit a normal distribution on `values` """
        num  = len(values)
        mean = sum(values)/num
        dm2  = sum((v-mean)**2 for v in values)
                
        #Dreturn [num,mean,dm2, values[:], 0, []]  #debug
        return [num,mean,dm2]

    @classmethod
    def add(cls, param, value):
        """ add `value` to the fitted `param`, **in-place** """
        param[0]  += 1               # number
        delta = value - param[1]
        param[1] += delta/param[0]   # mean
        param[2] += delta*(value - param[1])
        
        #Dparam[3].append(value)   #debug
        #Dparam[4] += 1            #debug

    @classmethod                                                
    def remove(cls, param, value):              
        """ remove `value` to the fitted `param`, **in-place** """
        #Dparam[3] = param[3][:]        #debug
        #Dparam[3].remove(value)        #debug
        #Dparam[4] += 1                 #debug
        #Dparam[5] = param[5]+[value]   #debug
        
        param[0] -= 1
        delta = value - param[1]
        param[1] -= delta/param[0]
        param[2] -= delta*(value - param[1])

    @classmethod
    def replace(cls, param, remove, add):
        """ remove values un `remove` and add those in `add`, **in-place** """
        for v in remove: cls.remove(param, v)
        for v in add:    cls.add(param, v)

    @classmethod
    def std(cls, param):
        """" return standard deviation of distribution `param` """
        return param[2]/(param[0]-1)  # dm2 / (n-1)
        
    @classmethod
    def pdf(cls, param, values):
        """ return probability of given `values` """
        num,mean,dm2 = param
        std  = dm2/(num-1)
        N = lambda x: math.exp(-.5*((x-mean)/std)**2) / (2.506628274631*std)
        return map(N, values) 


