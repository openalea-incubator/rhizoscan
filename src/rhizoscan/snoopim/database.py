"""
Some tool to read the computed "database" from a dataset
"""
from . import get_dataset_path

def read_img_ok(dataset_name):
    import csv
    from ast import literal_eval
    
    def eval_num(txt):
        try:
            return literal_eval(txt)
        except:
            return txt
    
    # read analysed image 
    f = get_dataset_path(dataset_name, 'Database','database', 'imgListOK.csv')
    with open(f, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        header  = dict()
        content = []                   
        for row in reader:
            row = [r.split('=') for r in row]
            row = dict((r[0],eval_num(r[1])) for r in row)
            if 'img' in row:
                content.append(row)
            else:
                header.update(row)
                
    return header, content

def description(name, verbose=False):
    """
    Return a description of database extracted from dataset `name`
    If verbose: print it nicely
    """
    header, content = read_img_ok(name)
    
    # compute global descriptor
    feature = [c.get('nb_desc',0) for c in content]
    n,tt,mi,ma = map(lambda f:f(feature), [len,sum,min,max])
    
    descr = list(header.iteritems())
    descr.append(('image_number' , n))
    descr.append(('feature_total', tt))
    descr.append(('feature_min'  , mi))
    descr.append(('feature_max'  , ma))
    descr.append(('feature_mean' , tt/n))

    if verbose:
        print '\n'.join(map(lambda x: x[0]+': '+str(x[1]),descr))
            
    return dict(descr)
    
def image_to_id_map(name):
    import os 
    
    header, content = read_img_ok(name)
    base_path = header.get('path','')
    
    i2i = dict()
    for c in content:
        i2i[os.path.join(base_path,c['img'])] = c['id']
    
    return i2i
