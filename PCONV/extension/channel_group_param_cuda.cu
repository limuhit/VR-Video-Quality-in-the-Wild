#include "channel_group_param.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void channel_group_param_opt::init(){
    init_base();
}

void channel_group_param_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    inner_shape_ = height_*width_;

}

void channel_group_param_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, height_, width_});
    reshape_top_base(option,shapes);
}


void channel_group_param_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, height_, width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void channel_group_param_forward_kernel(const int nthreads, const scalar_t* const input, 
    scalar_t* output, const int inner_shape, const int cin, const int cout, const int channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index / inner_shape;
        int pc = pt % channel;
        int pn = pt / channel;
        int gout = pn / cout;
        int cout = pc / cin;
        if(cout>=gout){
            output[index] = 0;
        }else{
            output[index] = input[index];
        }
    }
}

template <typename scalar_t>
__global__ void channel_group_param_forward_kernel_hidden(const int nthreads, const scalar_t* const input,  
    scalar_t* output, const int inner_shape, const int cin, const int cout, const int channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index / inner_shape;
        int pc = pt % channel;
        int pn = pt / channel;
        int gout = pn / cout;
        int cout = pc / cin;
        if(cout>gout){
            output[index] = 0;
        }else{
            output[index] = input[index];
        }
    }
}

std::vector<at::Tensor>  channel_group_param_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "channel_group_param_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    if(hidden_){
                        channel_group_param_forward_kernel_hidden<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), inner_shape_,cin_,cout_,channel_);

                    }else{
                        channel_group_param_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), inner_shape_,cin_,cout_,channel_);
                    }
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

std::vector<at::Tensor>  channel_group_param_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "channel_group_param_backward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    if(hidden_){
                        channel_group_param_forward_kernel_hidden<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_diff.data_ptr<scalar_t>(), bottom_diff_[0].data_ptr<scalar_t>(), inner_shape_,cin_,cout_,channel_);

                    }else{
                        channel_group_param_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_diff.data_ptr<scalar_t>(), bottom_diff_[0].data_ptr<scalar_t>(), inner_shape_,cin_,cout_,channel_);
                    }
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}

template <typename scalar_t>
__global__ void channel_group_param_forward_kernel_uneven(const int nthreads, const scalar_t* const input,  
    scalar_t* output, const int * param, const int inner_shape, const int cout, const int channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index / inner_shape;
        int pc = pt % channel;
        int pn = pt / channel;
        int gout = pn / cout;
        if(pc>=param[gout]){
            output[index] = 0;
        }else{
            output[index] = input[index];
        }
    }
}


std::vector<at::Tensor>  channel_group_param_opt::forward_cuda_uneven(at::Tensor  bottom_data, at::Tensor param) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "channel_group_param_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    channel_group_param_forward_kernel_uneven<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), param.data_ptr<int>(), inner_shape_, cout_, channel_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

std::vector<at::Tensor>  channel_group_param_opt::backward_cuda_uneven(at::Tensor  top_diff, at::Tensor param) 
{
    reshape_bottom(top_diff.options());
	int count;
    
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "channel_group_param_backward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    channel_group_param_forward_kernel_uneven<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, top_diff.data_ptr<scalar_t>(), bottom_diff_[0].data_ptr<scalar_t>(), param.data_ptr<int>(), inner_shape_, cout_, channel_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}