#include "three_d_texture_helper.cuh"

template <class DATA_T>
ThreeDTextureHelper<DATA_T>::ThreeDTextureHelper(int number, cudaStream_t stream)
  : TextureHelper<ThreeDTextureHelper<DATA_T>, DATA_T>(number, stream)
{
  cpu_values_.resize(number);
  layer_copy_.resize(number);
  for (std::vector<bool>& layer : layer_copy_)
  {
    // sets all current indexes to be true
    std::fill(layer.begin(), layer.end(), false);
  }
}

template <class DATA_T>
void ThreeDTextureHelper<DATA_T>::allocateCudaTexture(int index)
{
  TextureHelper<ThreeDTextureHelper<DATA_T>, DATA_T>::allocateCudaTexture(index);

  TextureParams<DATA_T>* param = &this->textures_[index];

  // TODO check to make sure our alloc is correct, i.e. extent is valid
  HANDLE_ERROR(cudaMalloc3DArray(&(param->array_d), &(param->channelDesc), param->extent));
}

template <class DATA_T>
void ThreeDTextureHelper<DATA_T>::updateTexture(const int index, const int z_index, std::vector<DATA_T>& values)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;
  int d = param->extent.depth;

  // check that the sizes are correct
  if (values.size() != w * h)
  {
    throw std::runtime_error(std::string("Error: invalid size to updateTexture ") + std::to_string(values.size()) +
                             " != " + std::to_string(w * h));
  }

  // TODO needs to be in the data format used for textures

  // copy over values to cpu side holder
  if (param->column_major)
  {
    for (int j = 0; j < w; j++)
    {
      for (int i = 0; i < h; i++)
      {
        int columnMajorIndex = j * h + i;
        int rowMajorIndex = (w * h * z_index) + i * w + j;
        cpu_values_[index][rowMajorIndex] = values[columnMajorIndex];
      }
    }
  }
  else
  {
    auto start = cpu_values_[index].begin() + (w * h * z_index);
    std::copy(values.begin(), values.end(), start);
  }
  // tells the object to copy it over next time that happens
  layer_copy_[index][z_index] = true;
  param->update_data = true;
}

template <class DATA_T>
void ThreeDTextureHelper<DATA_T>::updateTexture(
    const int index, const int z_index,
    const Eigen::Ref<const Eigen::Matrix<DATA_T, Eigen::Dynamic, Eigen::Dynamic>, 0,
                     Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>
        values)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  int w = param->extent.width;
  int h = param->extent.height;

  // check that the sizes are correct
  if (values.size() != w * h)
  {
    throw std::runtime_error(std::string("Error: invalid size to updateTexture ") + std::to_string(values.size()) +
                             " != " + std::to_string(w * h));
  }

  // TODO needs to be in the data format used for textures

  // copy over values to cpu side holder
  if (param->column_major)
  {
    for (int j = 0; j < w; j++)
    {
      for (int i = 0; i < h; i++)
      {
        int columnMajorIndex = j * h + i;
        int rowMajorIndex = (w * h * z_index) + i * w + j;
        cpu_values_[index][rowMajorIndex] = values.data()[columnMajorIndex];
      }
    }
  }
  else
  {
    auto start = cpu_values_[index].data() + (w * h * z_index);
    memcpy(start, values.data(), values.size() * sizeof(DATA_T));
  }
  // tells the object to copy it over next time that happens
  layer_copy_[index][z_index] = true;
  param->update_data = true;
}

// TODO update texture where everything is copied over in one go

template <class DATA_T>
__device__ DATA_T ThreeDTextureHelper<DATA_T>::queryTexture(const int index, const float3& point)
{
  // printf("query texture at (%f, %f, %f) = %f\n", point.x, point.y, point.z,
  //       tex3D<DATA_T>(this->textures_d_[index].tex_d, point.x, point.y, point.z));
  return tex3D<DATA_T>(this->textures_d_[index].tex_d, point.x, point.y, point.z);
}

template <class DATA_T>
bool ThreeDTextureHelper<DATA_T>::setExtent(int index, cudaExtent& extent)
{
  if (extent.depth == 0)
  {
    throw std::runtime_error(std::string("Error: extent in setExtent invalid,"
                                         " cannot use depth == 0 in 3D texture: using ") +
                             std::to_string(extent.depth));
  }

  if (!TextureHelper<ThreeDTextureHelper<DATA_T>, DATA_T>::setExtent(index, extent))
  {
    return false;
  }

  for (std::vector<DATA_T>& layer : cpu_values_)
  {
    layer.resize(extent.width * extent.height * extent.depth);
  }

  // TODO recopy better when depth changes if possible

  // this means we have changed our extent so we need to copy over all the data layers again
  for (std::vector<bool>& layer : layer_copy_)
  {
    // resizes the array to account for change in depth
    layer.resize(extent.depth);
    // sets all current indexes to be true
    std::fill(layer.begin(), layer.end(), true);
  }

  return true;
}

template <class DATA_T>
void ThreeDTextureHelper<DATA_T>::copyDataToGPU(int index, bool sync)
{
  TextureParams<DATA_T>* param = &this->textures_[index];
  auto copy_vec = layer_copy_[index].begin();

  int w = param->extent.width;
  int h = param->extent.height;
  int d = param->extent.depth;

  cudaMemcpy3DParms params = { 0 };
  params.srcPtr = make_cudaPitchedPtr(cpu_values_[index].data(), w * sizeof(DATA_T), w, h);
  params.dstArray = param->array_d;
  params.kind = cudaMemcpyHostToDevice;
  params.dstPos = make_cudaPos(0, 0, 0);
  params.srcPos = make_cudaPos(0, 0, 0);

  // TODO check if we just need to copy it all and do that

  // current index of z we are looking at
  int cur_z_index = 0;
  int prev_z_pos = -1;

  // TODO since we cannot have an extent of zero depth this is fine
  while (cur_z_index + 1 < d)
  {
    if (*(copy_vec + 1) and *copy_vec)
    {
      // if next index is true and cur is true keep building up copy
      copy_vec++;
      cur_z_index++;
      continue;
    }
    else if (!*(copy_vec + 1) and *copy_vec)
    {
      // if the next one is false and current is true begin a copy
      params.extent = make_cudaExtent(w, h, cur_z_index - prev_z_pos);

      HANDLE_ERROR(cudaMemcpy3DAsync(&params, this->stream_));
      prev_z_pos = cur_z_index;
    }
    else if (*(copy_vec + 1) and !*copy_vec)
    {
      // if the next one is true and cur is false start building up copy
      params.dstPos = make_cudaPos(0, 0, cur_z_index + 1);
      params.srcPos = make_cudaPos(0, 0, cur_z_index + 1);

      prev_z_pos = cur_z_index;
    }

    // increment counters
    copy_vec++;
    cur_z_index++;
  }

  // execute whatever copy is left
  if (prev_z_pos + 1 != cur_z_index)
  {
    params.extent = make_cudaExtent(w, h, cur_z_index - prev_z_pos);
    HANDLE_ERROR(cudaMemcpy3DAsync(&params, this->stream_));
  }

  for (std::vector<bool>& layer : layer_copy_)
  {
    std::fill(layer.begin(), layer.end(), false);
  }

  if (sync)
  {
    cudaStreamSynchronize(this->stream_);
  }
}
