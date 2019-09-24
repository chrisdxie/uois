#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/************************************************************
 hough voting layer
*************************************************************/
std::vector<at::Tensor> hough_voting_cuda_forward(
    at::Tensor label,
    at::Tensor directions,
    int skip_pixels,
    float inlierThreshold,
    int angle_discretization,
    int inlier_distance,
    float percentageThreshold,
    int object_center_kernel_radius);

std::vector<at::Tensor> hough_voting_forward(
    at::Tensor label,
    at::Tensor directions,
    int skip_pixels,
    float inlierThreshold,
    int angle_discretization, 
    int inlier_distance,
    float percentageThreshold,
    int object_center_kernel_radius)
{
  CHECK_INPUT(label);
  CHECK_INPUT(directions);

  return hough_voting_cuda_forward(label, directions, skip_pixels, 
        inlierThreshold, angle_discretization, inlier_distance, 
        percentageThreshold, object_center_kernel_radius);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hough_voting_forward", &hough_voting_forward, "hough_voting forward (CUDA)");
}
