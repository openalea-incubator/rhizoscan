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

4. change Ctrl+Shift+O/S to Ctrl+O/S
 - keep Ctrl+Shift+S for "save as"
 - Ctrl+S call "save as" when no filename is set
 - when "save as", add file to current project
 - when save and save as, call project.save 
 
 
Known bugs:
===========
 - some time, unset edge_type appears. I don't know how to reproduce this...
