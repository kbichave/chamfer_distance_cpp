// #include <torch/torch.h>
#include <pcl/io/ply_io.h>
#include "torch_cd/torch_cd.hpp"
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>


namespace po = boost::program_options;

bool fileExist(const std::string& name)
{
    std::ifstream f(name.c_str());  // New enough C++ library will accept just name
    return f.is_open();
}

bool processCommandLine(int argc, char** argv,
	std::string &fileCloud1,
	std::string &fileCloud2)
{
	try
	{
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Given a point cloud this code computes chamfer distance between the two.")
			("fileCloud1,f", po::value<std::string>(&fileCloud1)->required(), "Input first point cloud file in .ply format")
            ("fileCloud2,f", po::value<std::string>(&fileCloud2)->required(), "Input second point cloud file in .ply format")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);

		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			return false;
		}

		po::notify(vm);
	}
	catch (std::exception& e)
	{
		std::cerr << "ERROR: " << e.what() << "\n";
		return false;
	}
	catch (...)
	{
		std::cerr << "Unknown error!" << "\n";
		return false;
	}

	return true;
}


int main(int argc, char *argv[])
{   
    std::string data_file_1;
    std::string data_file_2;
    bool result = processCommandLine(argc, argv, data_file_1, data_file_2);
    if (!result)
        return 1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_2(new pcl::PointCloud<pcl::PointXYZ>);
    if (fileExist(data_file_1))
        pcl::io::loadPLYFile(data_file_1, *cloud_1);
    else
    {
        std::cout << "Point cloud file does not exsist or cannot be opened!!" << std::endl;
        return 1;
    }
    if (fileExist(data_file_2))
        pcl::io::loadPLYFile(data_file_2, *cloud_2);
    else
    {
        std::cout << "Point cloud file does not exsist or cannot be opened!!" << std::endl;
        return 1;
    }

    

    int num_points_cloud_1 = cloud_1->points.size();
    // # at::Tensor output = torch::from_blob(output_array, {batch_size, p_cloud_size, 3}).clone();
    int num_points_cloud_2 = cloud_2->points.size();
    std::vector<int> points_in_cloud = {num_points_cloud_1, num_points_cloud_2};


   
    std::vector<std::vector<float>> cloud_2_vector;
    for(int i=0; i<num_points_cloud_2; i++)
    {
        std::vector<float> tmp;
        tmp.push_back(cloud_2->points[i].x);
        tmp.push_back(cloud_2->points[i].y);
        tmp.push_back(cloud_2->points[i].z);
        cloud_2_vector.push_back(tmp);
    }
    
     std::vector<std::vector<float>> cloud_1_vector;
    for(int i=0; i<cloud_1->points.size(); i++)
    {
        std::vector<float> tmp;
        tmp.push_back(cloud_1->points[i].x);
        tmp.push_back(cloud_1->points[i].y);
        tmp.push_back(cloud_1->points[i].z);
        cloud_1_vector.push_back(tmp);
    }

 
    int k = compute_chamfer(cloud_1_vector, cloud_2_vector, points_in_cloud);
    return 1;
}


