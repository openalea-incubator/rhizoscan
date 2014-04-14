import os

image_file = 'test/data/pipeline/arabido.png'
project_file = 'test/data/pipeline/arabidopsis/database.ini'

def test_arabidopsis_pipeline():
    from rhizoscan.root.pipeline.arabidopsis import pipeline
    from rhizoscan.datastructure import Mapping
    
    filename = os.path.abspath(image_file)
    assert os.path.exists(filename), "could not find test image file:"+filename 
    
    d = Mapping(filename=filename, plant_number=2,
                fg_smooth=1, border_width=.08,leaf_height=[0,.4],root_max_radius=5)
    
    pipeline.run(namespace=d)
        
    assert d.has_key('image')   , "pipeline did not compute 'image'" 
    assert d.has_key('pmask')   , "pipeline did not compute 'pmask'"
    assert d.has_key('rmask')   , "pipeline did not compute 'rmask'"
    assert d.has_key('seed_map'), "pipeline did not compute 'seed_map'"
    assert d.has_key('graph')   , "pipeline did not compute 'graph'"
    assert d.has_key('tree')    , "pipeline did not compute 'tree'"
    
    import numpy as np
    t = d.tree
    assert t.axe.number>=8, "not the correct number of axes:"+str(t.axe.number)
    # problem: there is an axe with only a seed segment ??!
    assert (np.unique(t.axe.plant)==[0,1,2]).all(), "not the correct number of plants"
    
def test_load_dataset():
    from rhizoscan.tool.path import abspath
    from rhizoscan.root.pipeline.dataset import make_dataset
    
    pfile = os.path.abspath(project_file)
    pdir  = os.path.dirname(pfile)
    
    assert os.path.exists(pfile), "could not find test project file:"+pfile 
    
    def test_loaded_dataset(ds,invalid,out,exp_out):
        assert len(ds)==4, "invalid number of image file in test project"
        assert len(invalid)==1, "invalid number of invalid file in test project"
        assert hasattr(ds[0], 'filename'), "dataset item has not 'filename' attribute"
        assert hasattr(ds[0], 'metadata'), "dataset item has not 'metadata' attribute"
        assert hasattr(ds[0], '__map_storage__'), "dataset item has not output map storage set"
        
        assert out==exp_out

    ds, invalid, output = make_dataset(ini_file=pfile, out_dir='test_out', verbose=0)
    test_loaded_dataset(ds,invalid,output, os.path.join(pdir,'test_out'))

    ds, invalid, output = make_dataset(ini_file=pfile, verbose=0)
    test_loaded_dataset(ds,invalid,output, os.path.join(pdir,'outputs'))
        
    return ds
    


