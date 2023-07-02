#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class channel_group_param_opt: public base_opt{
	public:
		channel_group_param_opt(int ngroup, int cin, int cout, bool hidden, int device = 0, bool timeit=false){
			ngroup_ = ngroup;
			cin_ = cin;
			cout_ = cout;
			hidden_ = hidden;
			base_opt_init(device,timeit);
		}
		~channel_group_param_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
		void reshape_top(at::TensorOptions option);
		void reshape_bottom(at::TensorOptions option);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		std::vector<at::Tensor>  forward_cuda_uneven(at::Tensor  bottom_data, at::Tensor param);
		std::vector<at::Tensor>  backward_cuda_uneven(at::Tensor  top_diff, at::Tensor param);
		int ngroup_;
		int cin_;
		int cout_;
		bool hidden_;
		int inner_shape_;
};
