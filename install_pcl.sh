# Clone latest PCL
  apt-get update
  apt-get  install -y git

cd ~/Documents
# for clone pcl-1.8.1
# git clone --branch pcl-1.8.1 https://github.com/PointCloudLibrary/pcl.git pcl-trunk 
git clone https://github.com/PointCloudLibrary/pcl.git pcl-trunk
ln -s pcl-trunk pcl
cd pcl

# Install prerequisites
  apt-get  install -y g++
  apt-get  install -y cmake cmake-gui
  apt-get  install -y doxygen
  apt-get  install -y mpi-default-dev openmpi-bin openmpi-common
  apt-get  install -y libflann1.8 libflann-dev
  apt-get  install -y libeigen3-dev
  apt-get  install -y libboost-all-dev
  apt-get  install -y libvtk6-dev libvtk6.2 libvtk6.2-qt
#  apt-get  install -y libvtk5.10-qt4 libvtk5.10 libvtk5-dev  # I'm not sure if this is necessary.
  apt-get  install -y 'libqhull*'
  apt-get  install -y libusb-dev
  apt-get  install -y libgtest-dev
  apt-get  install -y git-core freeglut3-dev pkg-config
  apt-get  install -y build-essential libxmu-dev libxi-dev
  apt-get  install -y libusb-1.0-0-dev graphviz mono-complete
  apt-get  install -y qt-sdk openjdk-9-jdk openjdk-9-jre
  apt-get  install -y phonon-backend-gstreamer
  apt-get  install -y phonon-backend-vlc
  apt-get  install -y libopenni-dev libopenni2-dev
  apt-get  install -y libflann-dev

# Compile and  install -y PCL
mkdir release
cd release
cmake -DCMAKE_BUILD_TYPE=None -DBUILD_GPU=OFF -DBUILD_apps=ON -DBUILD_examples=ON ..
make -j 8
  make  install 


