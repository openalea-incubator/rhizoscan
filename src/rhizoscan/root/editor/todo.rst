TODO:
=====

1. move model edition action to model
 - edition registering of TreeModel into TreePresenter
 - try-except of edition by TreePresenter
 
 
2. generalized action registering
 - for any menu (add action attrib 'menu')
 - referencable by a key: which key, function or name or ...?
 - what about menu/action-list update
 - what about oa plugin 
 
3. controls?
 
Known bugs:
===========
 - reparent on another plants, if reparent is not done on branching
     * dont change plant of sub-axes
     * ----------- order -----------
     => subaxe parent don't change to created one !
 - reparent don't update color
