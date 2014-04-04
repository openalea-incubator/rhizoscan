

def image_io(format, ser_dtype, ser_scale):
    import os
    import tempfile
    import numpy as np
    from rhizoscan.image import Image
    
    fid, fname = tempfile.mkstemp(suffix='.'+format.lower())
    os.close(fid)
 
    img = Image(np.arange(10*18,dtype='uint8').reshape(10,18))
    img.set_serializer(pil_format=format, ser_dtype=ser_dtype, ser_scale=ser_scale)
    img.set_file(fname)
    
    loader = img.dump()
    img2 = loader.load()

    img.set_file(-1) # delete the file
    
    # assert serialization parameters are equivalent
    ser_param = ['img_dtype', 'img_scale', 'pil_format', 'pil_mode', 'pil_param', 'ser_color', 'ser_dtype', 'ser_scale']
    ser1 = img.get_serializer()
    ser2 = img2.get_serializer()
    
    assert ser2 is not None
    
    diff = [p for p in ser_param if getattr(ser1,p)<>None and getattr(ser1,p)<>getattr(ser2,p)]
    if len(diff)<>0:
        print [(getattr(ser1,p),getattr(ser2,p)) for p in diff]
        raise AssertionError('Image.serializer diff: ' + ','.join(diff))
    
    # assert the retrieved values are the same
    print np.array(img==img2)
    assert np.array(img==img2).all()
    
def test_IO_png_scale_dtype():
    image_io(format='PNG', ser_dtype=None, ser_scale='dtype')
def test_IO_png_scale_1():
    image_io(format='PNG', ser_dtype=None, ser_scale=1)
    
def test_IO_tiff_scale_dtype():
    image_io(format='TIFF', ser_dtype='f', ser_scale='dtype')
def test_IO_tiff_scale_1():
    image_io(format='TIFF', ser_dtype='f', ser_scale=1)
    

