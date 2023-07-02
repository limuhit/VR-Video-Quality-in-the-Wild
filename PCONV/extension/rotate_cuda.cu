#include "rotate.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void rotate_opt::init(){
    init_base();
}

bool rotate_opt::reshape(int num, int channel, int height, int width){
    return reshape_base(num, channel, height, width); 
}

void rotate_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_, channel_, height_, width_});
    shapes.push_back({num_, height_, width_, 3});
    shapes.push_back({num_, 9});
    shapes.push_back({num_, height_, width_, 3});
    shapes.push_back({num_, height_, width_, 2});
    reshape_top_base(option,shapes);
}

void rotate_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}

template <typename scalar_t>
__global__ void rt_init_xyz_kernel(int num, scalar_t * data, const int height, const int width, scalar_t hy, scalar_t hx, float pi){
    CUDA_KERNEL_LOOP(i, num) {
        int w = i % width;
        int h = (i / width) % height;
        float theta = (w + 0.5) / hx * 2 * pi - pi;
        float phi = 0.5*pi - (h+0.5) / hy * pi;
        data[i*3] = cos(phi)*cos(theta);
        data[i*3+1] = cos(phi)*sin(theta);
        data[i*3+2] = sin(phi);
    }
}


template <typename scalar_t>
__global__ void rotation_kernels(const int nthreads, const scalar_t* const theta_phi, scalar_t * const r){
	CUDA_KERNEL_LOOP(index, nthreads) {
		scalar_t a11,a12,a13,a21,a22,a23,a31,a32,a33;
		scalar_t b11,b12,b13,b21,b22,b23,b31,b32,b33;
		scalar_t c,s;
		a11 = cos(theta_phi[index*2]);
		a12 = -sin(theta_phi[index*2]);
		a13 = 0;
		a21 = -a12;
		a22 = a11;
		a23 = 0;
		a31 = 0;
		a32 = 0;
		a33 = 1;
		c = cos(theta_phi[index*2+1]);
		s = sin(theta_phi[index*2+1]);
		b11 = c + (1-c)*a12*a12;
		b12 = (1-c)*a12*a22;
		b13 = -s * a22;
		b21 = (1-c)*a12*a22;
		b22 = c + (1-c)*a22*a22;
		b23 =  s * a12;
		b31 =  s * a22;
		b32 = -s * a12;
		b33 = c;
		r[index*9+0] = b11*a11 + b12*a21 + b13*a31;
		r[index*9+1] = b11*a12 + b12*a22 + b13*a32;
		r[index*9+2] = b11*a13 + b12*a23 + b13*a33;
		r[index*9+3] = b21*a11 + b22*a21 + b23*a31;
		r[index*9+4] = b21*a12 + b22*a22 + b23*a32;
		r[index*9+5] = b21*a13 + b22*a23 + b23*a33;
		r[index*9+6] = b31*a11 + b32*a21 + b33*a31;
		r[index*9+7] = b31*a12 + b32*a22 + b33*a32;
		r[index*9+8] = b31*a13 + b32*a23 + b33*a33;
	}
}

template <typename scalar_t>
__global__ void rt_transpose_kernel(int num, const scalar_t * const x, const scalar_t * y, scalar_t * const z, const int m){
    CUDA_KERNEL_LOOP(i, num) {
        int tb = i / m;
        int tm = i % m;
        int base_x =  tb * m * 3;
        int base_y = tb * 3 * 3;
        float xa = x[base_x+tm*3];
        float xb = x[base_x+tm*3 + 1];
        float xc = x[base_x+tm*3 + 2];
        z[base_x+tm*3] = xa * y[base_y] + xb * y[base_y+3] + xc * y[base_y+6];
        z[base_x+tm*3 + 1] = xa * y[base_y+1] + xb * y[base_y+4] + xc * y[base_y+7];
        z[base_x+tm*3 + 2] = xa * y[base_y+2] + xb * y[base_y+5] + xc * y[base_y+8];
    }
}

template <typename scalar_t>
__global__ void rt_cal_xyz_kernel(int num, scalar_t * const xyz, scalar_t * tf,  scalar_t hx, scalar_t hy, float pi){
    CUDA_KERNEL_LOOP(i, num) {
        scalar_t lat = asin(xyz[i*3+2]);
        scalar_t tx = xyz[i*3];
        scalar_t ty = xyz[i*3+1];
        scalar_t theta = atan(ty/tx);
        if (tx<=0){
            if(ty>0){
                theta = theta + pi;
            }else{
                theta = theta - pi;
            }
        }
        tf[i*2] = (0.5 * theta / pi + 0.5) * hx - 0.5;
        tf[i*2+1] = (0.5 - lat / pi) * hy - 0.5;  
    }
}

template <typename scalar_t>
__global__ void rotate_forward_kernel(const int nthreads, const scalar_t* const input,  
    const scalar_t * tf, scalar_t * const output, const int inner_shape, const int hs, const int ws, const int channel) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ps = index % inner_shape;
        int tbase = index / inner_shape;
        int tn = tbase / channel;
        int base = tn*2*inner_shape;
        int tw = static_cast<int>(floor(tf[base + 2*ps]));
        int th = static_cast<int>(floor(tf[base + 2*ps + 1]));
        int ah = th > 0 ? th : 0;
        int bh = th + 1 >= hs ? hs-1 : th + 1;
        int aw = (tw + ws) % ws;
        int bw = (tw + 1) % ws;
        //int pw = (tw + 1) % ws;
        //int ph = th + 1 >= hs ?  hs-1 : th + 1; 
        scalar_t tx = tf[base + 2*ps] - tw;
        scalar_t ty = tf[base + 2*ps+1] - th;
        scalar_t ntx = 1. - tx;
        scalar_t nty = 1. - ty;
        output[index] = input[(tbase*hs+ah)*ws + aw]*ntx*nty + input[(tbase*hs+ah)*ws + bw]*tx*nty +  input[(tbase*hs+bh)*ws + aw]*ntx*ty + input[(tbase*hs+bh)*ws + bw]*tx*ty; 
    }
}


std::vector<at::Tensor>  rotate_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor theta_phi) 
{
    bool rp = reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "rotate_forward_cuda", 
			([&] {
                    scalar_t hx = width_;
                    scalar_t hy = height_;
                    if(rp){
                        count = num_* height_ * width_;
                        rt_init_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_>> >
                            (count, top_data_[1].data_ptr<scalar_t>(),height_,width_, hy, hx, pi_);
                        CUDA_POST_KERNEL_CHECK;
                    }
                    count = num_;
                    rotation_kernels<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                        (count, theta_phi.data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>());
                    CUDA_POST_KERNEL_CHECK;
                    count = num_ * channel_ * width_ * height_;
                    CUDA_POST_KERNEL_CHECK;
                    count = num_ * height_ * width_;
                    rt_transpose_kernel<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS, 0, stream_>>>
                            (count, top_data_[1].data_ptr<scalar_t>(), top_data_[2].data_ptr<scalar_t>(), 
                            top_data_[3].data_ptr<scalar_t>(), height_*width_);
                    
                    rt_cal_xyz_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, top_data_[3].data_ptr<scalar_t>(), top_data_[4].data_ptr<scalar_t>(), hx, hy, pi_);
                    count = num_ * channel_ * width_ * height_ ;
                    rotate_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[4].data_ptr<scalar_t>(), 
                            top_data_[0].data_ptr<scalar_t>(), height_*width_, height_, width_, channel_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void rotate_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        input[index] = output[index];
    }
}

std::vector<at::Tensor>  rotate_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "rotate_backward_cuda", 
			([&] {
                    
   			    }
			)
    );
    return bottom_diff_;
}