[ -d "/path/to/dir" ] && rm -rf build/
mkdir build
cd build
cmake ..
make
./chamfer_distance --fileCloud1 "../muffler_2.ply" --fileCloud2 "../muffler_2.ply"