[ -d "/path/to/dir" ] && rm -rf build/
mkdir build
cd build
cmake ..
make
./chamfer_distance --fileCloud1 <path_to_first_file> --fileCloud2 <path_to_second_file> 
