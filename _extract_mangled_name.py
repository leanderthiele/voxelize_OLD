
name = None
with open('./tmp_nm_output.txt', 'r') as f :
    for line in f : 
        line = line.split()
        for s in line :
            if 'voxelize' in s :
                name = s

if name is None :
    raise RuntimeError('did not find the mangled name.')
else :
    with open('./voxelize/MANGLEDNAME.txt', 'w') as f :
        f.write(name)
