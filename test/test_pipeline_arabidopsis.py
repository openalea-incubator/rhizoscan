import os


project_file = 'data/pipeline/arabidopsis/database.ini'
image_file   = 'data/pipeline/arabidopsis/J10/Photo_011.jpg'


def test_load_DB():
    from rhizoscan.root.pipeline.dataset import make_dataset
    pfile = os.path.abspath(project_file)
    
    assert os.path.exists(pfile)   # project file in absolute 
    
    db, invalid, output = make_dataset(ini_file=pfile, output='output', verbose=0)
    
    assert len(db)==4
    assert len(invalid)==1
    assert hasattr(db[0], 'filename')
    assert hasattr(db[0], 'metadata')
    assert hasattr(db[0], 'output')
    
    return db
    
def test_image_pipeline():
    from rhizoscan.root.pipeline import arabidopsis as pa
    import tempfile, os
    
    ifile  = os.path.abspath(image_file)
    outdir = tempfile.mkdtemp()
    
    try:
        # test pipeline with enforced modules computation 
        output = os.path.join(outdir, 'tmp_output_of_test_image_pipeline')
        pa.pipeline.run(image=ifile, output=output, plant_number=5, update=['all'])
        assert all([m.updated for m in pa.pipeline.modules])
        
        # test calling pipeline again: should not compute but reload data 
        pa.pipeline.run(image=ifile, output=output, update=[])
        assert not any([m.updated for m in pa.pipeline.modules])
    
    finally:
        # delete tmp folder
        import shutil
        shutil.rmtree(outdir)
