#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class channel_group_reshape_opt: public base_opt{
	public:
		channel_group_reshape_opt(int ngroup, int code_in, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			code_in_ = code_in;
			ipg_ = code_in_ / ngroup_;
			base_opt_init(device,timeit);
		}
		~channel_group_reshape_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		std::vector<at::Tensor>  forward_cuda_uneven(at::Tensor  bottom_data, at::Tensor param);
		std::vector<at::Tensor>  backward_cuda_uneven(at::Tensor  top_diff, at::Tensor param);
		int ngroup_;
		int code_in_;
		int cpg_;
		int inner_shape_, ipg_;
		at::Tensor gindex_;
		bool init_param_ = false;
};
