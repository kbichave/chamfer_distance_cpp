
#include <torch/torch.h>
#include "torch_cd.hpp"

void nnsearch(
    const int b, const int n, const int m,
    const float* xyz1,
    const float* xyz2,
    float* dist,
    int* idx)
{
    for (int i = 0; i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1[(i*n+j)*3+0];
            const float y1 = xyz1[(i*n+j)*3+1];
            const float z1 = xyz1[(i*n+j)*3+2];
            double best = 0;
            int besti = 0;
            for (int k = 0; k < m; k++) {
                const float x2 = xyz2[(i*m+k)*3+0] - x1;
                const float y2 = xyz2[(i*m+k)*3+1] - y1;
                const float z2 = xyz2[(i*m+k)*3+2] - z1;
                const double d=x2*x2+y2*y2+z2*z2;
                if (k==0 || d < best){
                    best = d;
                    besti = k;
                }
            }
            dist[i*n+j] = best;
            idx[i*n+j] = besti;
        }
    }
}


void chamfer_distance_forward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    const at::Tensor dist1, 
    const at::Tensor dist2, 
    const at::Tensor idx1, 
    const at::Tensor idx2) 
{
    const int batchsize = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* dist1_data = dist1.data<float>();
    float* dist2_data = dist2.data<float>();
    int* idx1_data = idx1.data<int>();
    int* idx2_data = idx2.data<int>();

    nnsearch(batchsize, n, m, xyz1_data, xyz2_data, dist1_data, idx1_data);
    nnsearch(batchsize, m, n, xyz2_data, xyz1_data, dist2_data, idx2_data);
}


void chamfer_distance_backward(
    const at::Tensor xyz1, 
    const at::Tensor xyz2, 
    at::Tensor gradxyz1, 
    at::Tensor gradxyz2, 
    at::Tensor graddist1, 
    at::Tensor graddist2, 
    at::Tensor idx1, 
    at::Tensor idx2) 
{
    const int b = xyz1.size(0);
    const int n = xyz1.size(1);
    const int m = xyz2.size(1);

    const float* xyz1_data = xyz1.data<float>();
    const float* xyz2_data = xyz2.data<float>();
    float* gradxyz1_data = gradxyz1.data<float>();
    float* gradxyz2_data = gradxyz2.data<float>();
    float* graddist1_data = graddist1.data<float>();
    float* graddist2_data = graddist2.data<float>();
    const int* idx1_data = idx1.data<int>();
    const int* idx2_data = idx2.data<int>();

    for (int i = 0; i < b*n*3; i++)
        gradxyz1_data[i] = 0;
    for (int i = 0; i < b*m*3; i++)
        gradxyz2_data[i] = 0;
    for (int i = 0;i < b; i++) {
        for (int j = 0; j < n; j++) {
            const float x1 = xyz1_data[(i*n+j)*3+0];
            const float y1 = xyz1_data[(i*n+j)*3+1];
            const float z1 = xyz1_data[(i*n+j)*3+2];
            const int j2 = idx1_data[i*n+j];

            const float x2 = xyz2_data[(i*m+j2)*3+0];
            const float y2 = xyz2_data[(i*m+j2)*3+1];
            const float z2 = xyz2_data[(i*m+j2)*3+2];
            const float g = graddist1_data[i*n+j]*2;

            gradxyz1_data[(i*n+j)*3+0] += g*(x1-x2);
            gradxyz1_data[(i*n+j)*3+1] += g*(y1-y2);
            gradxyz1_data[(i*n+j)*3+2] += g*(z1-z2);
            gradxyz2_data[(i*m+j2)*3+0] -= (g*(x1-x2));
            gradxyz2_data[(i*m+j2)*3+1] -= (g*(y1-y2));
            gradxyz2_data[(i*m+j2)*3+2] -= (g*(z1-z2));
        }
        for (int j = 0; j < m; j++) {
            const float x1 = xyz2_data[(i*m+j)*3+0];
            const float y1 = xyz2_data[(i*m+j)*3+1];
            const float z1 = xyz2_data[(i*m+j)*3+2];
            const int j2 = idx2_data[i*m+j];
            const float x2 = xyz1_data[(i*n+j2)*3+0];
            const float y2 = xyz1_data[(i*n+j2)*3+1];
            const float z2 = xyz1_data[(i*n+j2)*3+2];
            const float g = graddist2_data[i*m+j]*2;
            gradxyz2_data[(i*m+j)*3+0] += g*(x1-x2);
            gradxyz2_data[(i*m+j)*3+1] += g*(y1-y2);
            gradxyz2_data[(i*m+j)*3+2] += g*(z1-z2);
            gradxyz1_data[(i*n+j2)*3+0] -= (g*(x1-x2));
            gradxyz1_data[(i*n+j2)*3+1] -= (g*(y1-y2));
            gradxyz1_data[(i*n+j2)*3+2] -= (g*(z1-z2));
        }
    }
}


int compute_chamfer(std::vector<std::vector<float>> cloud_1_vector, std::vector<std::vector<float>>cloud_2_vector, std::vector<int> points_in_cloud)
{
    int batch_size = 1;
    int n = points_in_cloud[0];
    int m = points_in_cloud[1];

    // at::Tensor xyz1 = torch::randn({1,n, 3});
    // at::Tensor xyz2 = torch::randn({1,m, 3});

    int cloud_1_array [batch_size][points_in_cloud[0]][3];
    at::Tensor xyz1 = torch::from_blob(cloud_1_array, {batch_size, points_in_cloud[0], 3});
    for(int i=0; i<cloud_1_vector.size(); i++)
    {
        for(int j=0; j<3; j++)
        {
            xyz1[0][i][j] = cloud_1_vector[i][j];
        }
    }
    


    int cloud_2_array [batch_size][points_in_cloud[1]][3];
    at::Tensor xyz2 = torch::from_blob(cloud_2_array, {batch_size, points_in_cloud[1], 3});
    for(int i=0; i<cloud_2_vector.size(); i++)
    {
        for(int j=0; j<3; j++)
        {
            xyz2[0][i][j] = cloud_2_vector[i][j];
        }
    }
    
    at::Tensor dist1 = torch::zeros({batch_size, n});
    at::Tensor dist2 = torch::zeros({batch_size, m});
    at::Tensor idx1 = torch::zeros({batch_size, n}, at::kInt);
    at::Tensor idx2 = torch::zeros({batch_size, m}, at::kInt);
    chamfer_distance_forward(xyz1, xyz2, dist1, dist2, idx1, idx2);
    // // std::vector<float> dist1_vector = dist1[0].flat<float>().data();
    // std::cout<<"dist1"<<dist1;
    at::Tensor distance1 = torch::mean(dist1);
    at::Tensor distance2 = torch::mean(dist2);
    at::Tensor distance = distance1 + distance2;
    std::cout<< "Chamfer Distance: "<<(distance);
    return 1;
    
}