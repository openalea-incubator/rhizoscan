todo list in rhizoscan project
==============================


Make test functions for all modules/functions
---------------------------------------------
  - check code covering of current test


Remove openalea import from workflow.openalea
---------------------------------------------
  - make a function that create the workflow list 
  - move FuncFactory in new module (to wf.oa_factory?)
  - or finish oa autoloader
  - replace import test by something fast, such as:
       import sys
       len(filter(lambda p: 'openalea' in p and 'core' in p, sys.path))>0
  - or use setup.py-entry_points ?
   

Integrate/clarify dataset management by pipeline
------------------------------------------------
  - pipeline.run add argument load=False*/True/attempt & dump=False
     - load: namespace.load(attempt=True)
     - dump: only id hasattr(ns,'dump')??

  - pipeline.map( ..., load=True, ?dump=True):
     - if dump=False: merge returned output
   
   
Add dataset to datastructure
----------------------------
   - `[GLOBAL]` instead of `[global]`
   - `[GLOBAL] xxxx=.....$(filename)....`: value from parsing
   - `[GLOBAL] xxxx=.....$(base_dir/out_dir/...)....`
   - `[GLOBAL] xxxx=module.SomeClass(....)` ?

   - don't use metadata subtree?
      - implies a lots of possible changes/bugs...
      - loader_fields: all fields created by DatasetItem()

   #- `[PARSING] filename|...` all are parsed
   #- but _data_dir, _out_dir
   #    => or not, what to do with multiple parsed fields???
   #       if needed, add `name=alternative2filename`
  
      
Generalize serializer:
----------------------
   - declare serializer fct in entry points
      - w.r.t extension(?)
   
      
Clarify Data/Mapping & FileObject/FileStorage container system
--------------------------------------------------------------
   - ?? Data related to FileObject
   - Mapping related to FileStorage
      - FileStorage is also a FileObject?
      
   - move all contained switch to storage side
   
   - make it work for any subdir of Mapping file directory      
      
   
Tracking pipeline
-----------------
   - tracking node: declare (optional) arg that are taken from t-1
      - node(..., tracking=dict(arg_name=ns_t-1_name,...))
      
   - pipeline.track(ds, ....)
   - pipeline.append(init_node=..., trk_node=...)???
      => not sure, unique node imply unique set of inputs/outputs
      
      
Split non root package to another package 
-----------------------------------------
   - pkg name? jusci,... ?

