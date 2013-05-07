import os


project_file = 'data/pipeline/arabidopsis/project.ini'
image_file   = 'data/pipeline/arabidopsis/J10/Photo_011.jpg'


def test_load_DB():
    from rhizoscan.root.pipeline.database import parse_image_db
    pfile = os.path.abspath(project_file)
    
    assert os.path.exists(pfile)   # project file in absolute 
    
    db, invalid, output = parse_image_db(ini_file=pfile, output='output', verbose=0)
    
    assert len(db)==4
    assert len(invalid)==1
    assert hasattr(db[0], 'filename')
    assert hasattr(db[0], 'metadata')
    assert hasattr(db[0], 'output')
    
    return db
    
def test_image_pipeline():
    from rhizoscan.root.pipeline import arabidopsis as pa
    ifile = os.path.abspath(image_file)
    
    # test with modules computation enforced 
    pa.pipeline.run(image=ifile, output='~/tmp/pipeline_arabidopsis_test', plant_number=5, update=['all'])
    assert all([m.updated for m in pa.pipeline.modules])
    
    # test with modules computation enforced 
    pa.pipeline.run(image=image_file, output='~/tmp/pipeline_arabidopsis_test', update=[])
    assert not any([m.updated for m in pa.pipeline.modules])

