from rhizoscan.root.pipeline.database import parse_image_db
import os


project_file = 'data/arabidopsis_1/database.ini'

def load_DB():
    pfile = os.path.abspath(project_file)
    
    assert os.path.exists(pfile)   # project file in absolute 
    
    db, invalid, output = parse_image_db(ini_file=pfile, output='output', verbose=0)
    
    assert len(db)==4
    assert len(invalid)==1
    assert hasattr(db[0], 'filename')
    assert hasattr(db[0], 'metadata')
    assert hasattr(db[0], 'output')
    
    return db
    

