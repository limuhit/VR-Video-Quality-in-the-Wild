#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class rotate_opt: public base_opt{
	public:
		rotate_opt(int interp, int device = 0, bool timeit=false){
			interp_ = interp;
			pi_ = acos(-1.0);
			base_opt_init(device,timeit);
		}
		~rotate_opt(){}
		void init();
		bool reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, at::Tensor theta_phi);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int interp_;
		float pi_;
};
