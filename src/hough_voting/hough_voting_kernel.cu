#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <cfloat>
#include <time.h>
#include <thrust/extrema.h>
#include <Eigen/Geometry> 
#include <cublas_v2.h>

#define DIRECTIONS_CHANNELS 2

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


// Compute 2D Euclidean distance between [x,y] and [u,v]
__device__ inline float distance(int x, int y, int u, int v)
{
  float dx = x - u;
  float dy = y - v;
  return sqrt(dx * dx + dy * dy);
}

// Compute the cosine similarity between [cx,cy] - [x,y], and [u,v]
__device__ inline float angle_distance(int cx, int cy, int x, int y, float u, float v)
{
  float dx = cx - x;
  float dy = cy - y;
  float n1 = sqrt(u * u + v * v);
  float n2 = sqrt(dx * dx + dy * dy);
  float dot = u * dx + v * dy;
  float distance = dot / (n1 * n2);

  return distance;
}

// Compute the arrays kernel, which gives a list of foreground pixel locations
__global__ void compute_arrays_kernel(const int nthreads, const int* labelPointer,
    int* arrays, int* array_size) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int cls = labelPointer[index];
    if (cls > 0)
    {
      int offset = atomicAdd(array_size, 1);
      arrays[offset] = index;
    }
  }
}

// Compute dot product of [cx,cy] - [x,y] with [0, 1]. Discretize the angle into a bin
__device__ inline int discretization_angle_index(int cx, int cy, int x, int y, 
    int angle_discretization)
{
  float cos_sim = .5 * (angle_distance(cx, cy, x, y, 0., 1.) + 1); // in range [0,1]
  int angle_index = static_cast<int>(floor(cos_sim * (float) angle_discretization));
  if (angle_index == angle_discretization) {
    angle_index--; 
  }
  return angle_index;
}

// Compute the hough map
__global__ void compute_hough_kernel(const int nthreads, float* houghDirectionsPointer,
    const float* directionsPointer, int* arrays, int* array_size, const int height, 
    const int width, const float inlierThreshold, const int skip_pixels, const int angle_discretization,
    int inlier_distance) 
{

  int size = *array_size;

  CUDA_1D_KERNEL_LOOP(index, nthreads) // index is a pixel
  {
    // (cx, cy) is an element in the hough space. this is the index of the 2D matrix
    int cx = index % width;
    int cy = index / width;

    for (int i = 0; i < size; i += skip_pixels)
    {
      int offset = i;
      int location = arrays[offset];
      int x = location % width;
      int y = location / width;

      // read the direction
      offset = y * width + x;
      float v = directionsPointer[offset];
      offset = height * width + y * width + x;
      float u = directionsPointer[offset];

      // vote. Compute whether the pixel points to (cx, cy) and how close it is
      if (angle_distance(cx, cy, x, y, u, v) > inlierThreshold && 
          distance(cx, cy, x, y) < inlier_distance)
      {
        // Compute discretization angle
        int angle_index = discretization_angle_index(cx, cy, x, y, angle_discretization);
        houghDirectionsPointer[angle_index * height * width + index] = 1;
      }
    }
  }
}

// Compute the object center pixel locations
__global__ void compute_object_center_indices_kernel(const int nthreads, int* objectCenterIndicesPointer, 
  int* numObjectsPointer, int max_objects_to_consider, float* houghDirectionsPointer, int height, int width, 
  float percentageThreshold, int object_center_kernel_radius)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // (cx, cy) is an element in the hough space. this is the index of the 2D matrix
    int cx = index % width;
    int cy = index / width;

    int max_index = -1;

    if (houghDirectionsPointer[index] > percentageThreshold)
    {
      // check if the location is local maximum
      int is_local_max = 1;
      for (int x = cx - object_center_kernel_radius; x <= cx + object_center_kernel_radius; x++)
      {
        for (int y = cy - object_center_kernel_radius; y <= cy + object_center_kernel_radius; y++)
        {
          if (x >= 0 && x < width && y >= 0 && y < height)
          {
            if (houghDirectionsPointer[y * width + x] > houghDirectionsPointer[index])
            {
              is_local_max = 0;
              break;
            }

            // tie breaking
            if (houghDirectionsPointer[y * width + x] == houghDirectionsPointer[index] &&
                y * width + x > index)
            {
              is_local_max = 0;
              break;
            }
          }
        }
      }
      // add the location to object_center_indices
      if (is_local_max == 1 && (max_index + 1) < max_objects_to_consider)
      {
        max_index = atomicAdd(numObjectsPointer, 1);
        objectCenterIndicesPointer[max_index] = index;
      }
    }
  }
}

// Compute the initial masks
__global__ void compute_initial_masks_kernel(const int nthreads, int* initialMasksPointer,
    int* objectCenterIndicesPointer, int* numObjectsPointer, const float* directionsPointer, 
    int* arrays, const int height, const int width, const float inlierThreshold) 
{

  int num_object_centers = *numObjectsPointer;

  CUDA_1D_KERNEL_LOOP(index, nthreads) // index is a foreground pixel location
  {

    // Get (x,y) pixel location
    int location = arrays[index];
    int x = location % width;
    int y = location / width;

    // read the direction
    int offset = y * width + x;
    float v = directionsPointer[offset];
    offset = height * width + y * width + x;
    float u = directionsPointer[offset];

    // Keep track of closest center
    int closest_center_index = -1;
    float closest_center_distance = 1.0e6; // Something ridiculously large to start with

    for (int i = 0; i < num_object_centers; i++)
    {

      // Get (cx, cy) object center
      int cx = objectCenterIndicesPointer[i] % width;
      int cy = objectCenterIndicesPointer[i] / width;

      float dist_to_center = distance(cx, cy, x, y);
      float dist_in_angle = angle_distance(cx, cy, x, y, u, v);
      if (dist_in_angle > inlierThreshold &&
          dist_to_center < closest_center_distance)
      {
        closest_center_index = i;
        closest_center_distance = dist_to_center;
      }
    }

    if (closest_center_index > -1) // We chose a center
    {
      initialMasksPointer[closest_center_index * height * width + location] = 1;
    }
  }
}

std::vector<at::Tensor> hough_voting_cuda_forward(
    at::Tensor label,
    at::Tensor directions,
    int skip_pixels,
    float inlierThreshold,
    int angle_discretization,
    int inlier_distance,
    float percentageThreshold,
    int object_center_kernel_radius)
{
  const int kThreadsPerBlock = 1024;
  cudaError_t err;

  const int batch_size = directions.size(0);
  const int height = directions.size(2);
  const int width = directions.size(3);
  int num_pixels = height * width;


  // Create initial center masks
  int max_objects_to_consider = 50; // Cap number of potential objects to 50
  auto initial_masks = at::zeros({batch_size, max_objects_to_consider, height, width}, label.options());
  auto object_center_indices = at::zeros({batch_size, max_objects_to_consider}, label.options());
  auto num_objects = at::zeros({batch_size}, label.options());

  for (int batch_index = 0; batch_index < batch_size; batch_index++)
  {
    // Get all the pointers for this batch
    const int* labelPointer = label.data<int>() + batch_index * height * width;
    const float* directionsPointer = directions.data<float>() + batch_index * height * width * DIRECTIONS_CHANNELS;


    // step 1: compute a label index array. Keeps track of all foreground pixels
    auto arrays = at::zeros({height * width}, label.options()); // this array is a list of foreground pixel indices
    auto array_sizes = at::zeros({1}, label.options()); // how many pixels vote for foreground?
    compute_arrays_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      num_pixels, labelPointer, arrays.data<int>(), array_sizes.data<int>()); // fills those pixel arrays
    cudaThreadSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute label index: %s\n", cudaGetErrorString( err ) ); // fprintf can be used from CPU. prints to terminal (not jupyter notebook)
      exit( -1 );
    }


    // step 2: compute the hough directions
    auto hough_directions_frame = at::zeros({angle_discretization, height, width}, directions.options());
    compute_hough_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      num_pixels, hough_directions_frame.data<float>(), directionsPointer, arrays.data<int>(), 
      array_sizes.data<int>(), height, width, inlierThreshold, skip_pixels, angle_discretization, 
      inlier_distance);
    hough_directions_frame = hough_directions_frame.sum(/*dim=*/0) / angle_discretization; // Divide to get a percentage. Shape: [H x W]
    cudaThreadSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute hough space: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }


    // step 3: find the local maximums in hough directions
    auto object_center_indices_frame = at::zeros({max_objects_to_consider}, label.options());
    auto num_objects_frame = at::zeros({1}, label.options());
    compute_object_center_indices_kernel<<<(num_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      num_pixels, object_center_indices_frame.data<int>(), num_objects_frame.data<int>(), max_objects_to_consider, 
      hough_directions_frame.data<float>(), height, width, percentageThreshold, object_center_kernel_radius);
    cudaThreadSynchronize();

    // Check for errors
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute maximum: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }

    // Copy the object centers for this frame to the tensor
    cudaMemcpy(object_center_indices.data<int>() + batch_index * max_objects_to_consider, object_center_indices_frame.data<int>(), 
               max_objects_to_consider * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(num_objects.data<int>() + batch_index, num_objects_frame.data<int>(), 
               sizeof(int), cudaMemcpyDeviceToDevice);


    // step 4: Get initial masks for each object center
    auto initial_masks_frame = at::zeros({max_objects_to_consider, height, width}, label.options());
    int num_foreground_pixels, num_objects_host;
    cudaMemcpy(&num_foreground_pixels, array_sizes.data<int>(), sizeof(int), cudaMemcpyDeviceToHost); // Copy array_sizes from GPU memory to CPU memory

    if (num_foreground_pixels > 0) // Calling CUDA_1D_KERNEL_LOOP with nthreads = 0 poops out
    {
      compute_initial_masks_kernel<<<(num_foreground_pixels + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
        num_foreground_pixels, initial_masks_frame.data<int>(), object_center_indices_frame.data<int>(), 
        num_objects_frame.data<int>(), directionsPointer, arrays.data<int>(), height, width, inlierThreshold);
      cudaThreadSynchronize();
    }

    // Check for errors
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed compute initial masks: %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }

    // Copy the initial masks for this frame to the tensor
    cudaMemcpy(initial_masks.data<int>() + batch_index * max_objects_to_consider * height * width, initial_masks_frame.data<int>(), 
               max_objects_to_consider * height * width * sizeof(int), cudaMemcpyDeviceToDevice);

  }

  return {initial_masks, num_objects, object_center_indices};

}

