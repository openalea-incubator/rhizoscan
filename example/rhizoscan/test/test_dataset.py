"""
Test for rhizoscan.root.pipeline.dataset.Dataset
"""

correct_len = 6
def test_Dataset_constructor():
    from rhizoscan.datastructure  import Mapping
    from rhizoscan.root.pipeline.dataset import Dataset

    ds = Dataset(Mapping(a=i/2,b=i/3,__key__=i, sub=Mapping(item=0)) for i in range(correct_len))
    assert len(ds)==correct_len, 'incorrect item number (%d/%d)' % (len(ds),correct_len)
    return ds

def test_Dataset_keys():
    ds = test_Dataset_constructor()
    keys = range(correct_len)
    assert ds.keys()==keys, 'incorrect Dataset keys: '+str(ds.keys())+' instead of '+str(keys)

def test_Dataset_group_by():
    ds  = test_Dataset_constructor()
    cds = ds.group_by('a')
    assert cds.keys()==range(3), 'failed single key group_by'
    cds = ds.group_by(['a','b'])
    assert cds.keys()==[(1, 0), (0, 0), (1, 1), (2, 1)], 'failed multi-key group_by'

def test_Dataset_get_column():
    ds  = test_Dataset_constructor()
    col = ds.get_column('__key__')
    assert col==range(len(ds)), 'failed single key get_column'
    col = ds.get_column('sub.item')
    assert col==[0]*len(ds), 'failed multi-key get_column'
