// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void subAndDot(const int N, const int len, const Dtype* a, const Dtype* b, Dtype* out) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N){
        // int index = i%len;
        Dtype tmp = a[i] - b[i%len];
        out[i] = tmp*tmp;
    }
}

template <typename Dtype>
void OffTripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //LOG(INFO) << "enter triplet Forward_gpu";
    //LOG(INFO) << "inner_num_: " << inner_num_ << ", label_separator_: " << label_separator_;
    Dtype loss = Dtype(0);
    int batch_size = bottom[0]->num(); // get the batch_size
    //CHECK_EQ(batch_size, bottom[1]->count());
    //LOG(INFO) << batch_size << ":" << bottom[1]->num();
    //CHECK_EQ(batch_size, bottom[1]->count());
    CHECK_EQ(inner_num_, 128);
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* sub = diff_.mutable_gpu_data();
    Dtype* diff = diff_.mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(), Dtype(0.0), diff);

    vector<int> label_pos(batch_size, 0);
    vector<int> label_neg(batch_size, 0);
    for (int i = 0; i < batch_size; ++i)
    {
      int label = static_cast<int>(bottom_label[i]);
      label_pos[i] = label%1000*inner_num_;
      label /= 1000;
      label_neg[i] = label%1000*inner_num_;
    }

    for (int i = 0; i < batch_size; ++i)
    {
      Dtype res = alpha_;
      Dtype tmp = 0.0;
      // replace my own kernel function subAndDot
      caffe_gpu_sub(inner_num_, bottom_data + i*inner_num_, bottom_data + label_pos[i], sub);
      caffe_gpu_dot(inner_num_, sub, sub, &tmp);
      res += tmp;
      caffe_gpu_sub(inner_num_, bottom_data + i*inner_num_, bottom_data + label_neg[i], sub);
      caffe_gpu_dot(inner_num_, sub, sub, &tmp);
      res -= tmp;
      if (res > 0.0) {
        loss += res;
        caffe_gpu_sub(inner_num_, bottom_data + label_neg[i], bottom_data + label_pos[i], diff + i*inner_num_);
      }
    }

    top[0]->mutable_cpu_data()[0] = loss;///(Dtype(2)*bottom[0]->num());
}

template <typename Dtype>
void OffTripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        if (top[0]->cpu_diff()[0] != Dtype(1.0)){
            LOG(INFO) << "Triplet.cu top cpu_diff is not 1.0 is " << top[0]->cpu_diff()[0];
        }

        Dtype scale = Dtype(2.0)*top[0]->cpu_diff()[0]/bottom[0]->num();
        caffe_gpu_scale(
            bottom[0]->count(),    // count
            scale,                 // scale
            diff_.gpu_diff(),      // input
            bottom[0]->mutable_gpu_diff() // output
        );

	/*
	const Dtype* ptr = bottom[0]->cpu_diff();
	for (int i = 0; i < bottom[0]->num(); i++) {
	    int tmp = i*128;
	    std::cout << i << ": ";
	    for (int j = 0; j < 128; j++) {
	    	std::cout << ptr[tmp++] << " ";
	    }
	    std::cout << "\n";
	}
	
	bottom[0]->gpu_diff();
	CHECK_EQ(1,2);
	*/
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT OffTripletLossLayer::Backward_cpu" << std::endl;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(OffTripletLossLayer);
} // namespace caffe
