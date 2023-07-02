#include "channel_group_reshape.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void channel_group_reshape_opt::init(){
    init_base();
}

void channel_group_reshape_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    cpg_ = channel_ / ngroup_;
    inner_shape_ = height_*width_;
}

void channel_group_reshape_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*code_in_*inner_shape_, cpg_});
    reshape_top_base(option,shapes);
}

void channel_group_reshape_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void channel_group_reshape_forward_kernel(const int nthreads, const scalar_t* const input,  
     scalar_t * const output, const int inner_shape, const int code_channel, const int channel, 
     const int outer_shape, const int cpg, const int ipg) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pidx = index % outer_shape;
        int ci = index / outer_shape;
        int ps = pidx % inner_shape;
        int pnc = pidx / inner_shape;
        int pc = pnc % code_channel;
        int pn = pnc / code_channel;
        int pg = pc / ipg;
        int oidx = pidx * cpg + ci;
        int iidx = (pn*channel + pg * cpg + ci)*inner_shape + ps;
        output[oidx] = input[iidx];
    }
}


std::vector<at::Tensor>  channel_group_reshape_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "channel_group_reshape_forward_cuda", 
			([&] {
                    int outer_shape = num_ * code_in_ * inner_shape_;
                    count =  outer_shape * cpg_;
                    channel_group_reshape_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), inner_shape_, code_in_, 
                            channel_, outer_shape, cpg_, ipg_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void channel_group_reshape_backward_kernel(const int nthreads, scalar_t* const input,  
const scalar_t * const output, const int inner_shape, const int code_channel, const int channel, 
const int outer_shape, const int cpg, const int ipg) {
CUDA_KERNEL_LOOP(index, nthreads) {
        int ps =  index % inner_shape;
        int pnc = index / inner_shape;
        int pc = pnc % channel;
        int pn = pnc / channel;
        int pg = pc / cpg;
        int ci = pc % cpg;
        int obase =  ((pn*code_channel + pg * ipg)*inner_shape + ps)*cpg + ci;
        scalar_t tsum = 0;
        for(int i = 0; i< ipg; i++){
            tsum += output[obase+i*outer_shape];
        }
        input[index] = tsum;
    }
}


std::vector<at::Tensor>  channel_group_reshape_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "channel_group_reshape_backward_cuda", 
			([&] {
                    int outer_shape = inner_shape_ * cpg_;
                    count = num_ * channel_ * inner_shape_;
                    channel_group_reshape_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), inner_shape_, code_in_, channel_, 
                            outer_shape, cpg_, ipg_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}

template<class T>
struct SharedMemory
{
    __device__ inline operator  T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

template <int blockSize>
__global__ void produce_group_index(const int num, const int * param, int * gindex, const int ngroup){
    int * sdata = SharedMemory<int>();
    int tid = threadIdx.x;
    if(tid < ngroup+1){
        sdata[tid] = param[tid];
    }
    __syncthreads();
    for(int pid = tid; pid<num; pid+=blockSize){
        int idx = 0;
        for(; idx < ngroup; idx++){
            if(pid<sdata[idx+1]){
                break;
            }
        }
        gindex[pid] = idx;
        //printf("channel %d, group %d",pid,idx);
    }
}

template <typename scalar_t>
__global__ void channel_group_reshape_forward_kernel_uneven(const int nthreads, const scalar_t* const input,  const int * gindex,
     scalar_t * const output, const int inner_shape, const int code_channel, const int channel, 
     const int outer_shape, const int cpg, const int ipg) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pidx = index % outer_shape;
        int ci = index / outer_shape;
        int ps = pidx % inner_shape;
        int pnc = pidx / inner_shape;
        int pc = pnc % code_channel;
        int pn = pnc / code_channel;
        int pg = gindex[pc];
        int oidx = pidx * cpg + ci;
        int iidx = (pn*channel + pg * cpg + ci)*inner_shape + ps;
        output[oidx] = input[iidx];
    }
}

std::vector<at::Tensor>  channel_group_reshape_opt::forward_cuda_uneven(at::Tensor  bottom_data, at::Tensor param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
    if(!init_param_){
        gindex_ = at::zeros({code_in_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
        produce_group_index<1024><< <1, 1024, 1024*sizeof(int), stream_ >> >
            (code_in_,param.data_ptr<int>(),gindex_.data_ptr<int>(),ngroup_);
        init_param_ = true;
    }
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "channel_group_reshape_forward_cuda", 
			([&] {
                    int outer_shape = num_ * code_in_ * inner_shape_;
                    count =  outer_shape * cpg_;
                    channel_group_reshape_forward_kernel_uneven<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), gindex_.data_ptr<int>(), top_data_[0].data_ptr<scalar_t>(), 
                            inner_shape_, code_in_, channel_, outer_shape, cpg_, ipg_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void channel_group_reshape_backward_kernel_uneven(const int nthreads, scalar_t* const input, const int * param,
const scalar_t * const output, const int inner_shape, const int code_channel, const int channel, 
const int outer_shape, const int cpg, const int ipg) {
CUDA_KERNEL_LOOP(index, nthreads) {
        int ps =  index % inner_shape;
        int pnc = index / inner_shape;
        int pc = pnc % channel;
        int pn = pnc / channel;
        int pg = pc / cpg;
        int ci = pc % cpg;
        int obase =  (pn*code_channel*inner_shape + ps)*cpg + ci;
        scalar_t tsum = 0;
        for(int i = param[pg]; i< param[pg+1]; i++){
            tsum += output[obase+i*outer_shape];
        }
        input[index] = tsum;
    }
}


std::vector<at::Tensor>  channel_group_reshape_opt::backward_cuda_uneven(at::Tensor  top_diff, at::Tensor param) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "channel_group_reshape_backward_cuda", 
			([&] {
                    int outer_shape = inner_shape_ * cpg_;
                    count = num_ * channel_ * inner_shape_;
                    channel_group_reshape_backward_kernel_uneven<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(),  param.data_ptr<int>(), top_diff.data_ptr<scalar_t>(), 
                            inner_shape_, code_in_, channel_, outer_shape, cpg_, ipg_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}