"""
simple and practical decorators

.. currentmodule:: rhizoscan.misc.decorators

"""

__all__ = ['_property','class_or_instance_method', 'static_or_instance_method']

import types
class _property(property): pass  # property decorator without the property doc

class class_or_instance_method(object):
    """
    Decorator that makes a method act as either class or instance method 
    depending on the call. 
    
    :Example:
        >>> class A:
        >>>     @class_or_instance_method
        >>>     def func(cls_or_self):
        >>>         print cls_or_self
        >>> 
        >>> A.func()    # =>  print (class) A
        >>> A().func()  # =>  print A instance
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        inst_cls = owner if instance is None else instance
        ##return functools.partial(self.func, inst_cls)
        ##return new.instancemethod(self.func,inst_cls,owner)
        return types.MethodType(self.func,inst_cls)
        
class static_or_instance_method(object):
    """
    Decorator that makes a method act as either a static or instance method
    depending on the call.
    
    :Example:
        >>> class A:
        >>>     @static_or_instance_method
        >>>     def func(self_or_value):
        >>>         print self_or_value
        >>> 
        >>> A().func()  # =>  print A instance
        >>> A.func(42)  # =>  print 42
    """
    ##TODO: static_or_instance doc: decorator example
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner):
        if instance is None: 
            func = self.func
        else:
            func = types.MethodType(self.func,instance)
            #func = new.instancemethod(self.func,instance,instance.__class__)
            #func = functools.partial(self.func,instance)
            #func.__doc__ = self.func.__doc__ 
        return func

