g++ -O4 -ffast-math -std=c++17 -fopenmp \
  -Wall -Wextra -DNDEBUG \
  -shared -fPIC \
  -o libvoxelize.so voxelize.cpp

echo $(pwd) > voxelize/PATHTOSO.txt
echo $(nm -D libvoxelize.so) > tmp_nm_output.txt
python _extract_mangled_name.py
rm tmp_nm_output.txt

pip install . --user
