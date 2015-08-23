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
void TripletLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype loss = Dtype(0);
    int batch_size = bottom[0]->num(); // get the batch_size
    CHECK_EQ(batch_size, bottom[1]->count());

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* diff_mutable = diff_.mutable_gpu_data();
    Dtype* sub_mutable = sub_.mutable_gpu_data();
    Dtype* diff_diff = diff_.mutable_gpu_diff(); // store the diff
    caffe_set(diff_.count(), Dtype(0), diff_mutable);
    caffe_set(diff_.count(), Dtype(0), diff_diff);
    // #program
    vector<int> labels(batch_size, 0);
    for (int i = 0; i < batch_size; i++){
        labels[i] = static_cast<int>(bottom_label[i]);
    }

    int count = diff_.count();
    Dtype** mat = new Dtype*[batch_size];

    Dtype* device_scalar;
    Dtype* device_tmp;
    CUDA_CHECK(cudaMalloc((void**)&device_scalar, inner_num_*sizeof(Dtype)));
    CUDA_CHECK(cudaMalloc((void**)&device_tmp, batch_size*sizeof(Dtype)));
    caffe_gpu_set(batch_size, Dtype(1.0), device_scalar);

    for (int i = 0; i < batch_size; i++){
        int label = labels[i];
        mat[i] = new Dtype[batch_size];
        if (label < label_separator_) {
            subAndDot<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(batch_size, inner_num_,
            bottom_data, bottom_data+i*inner_num_, sub_mutable);

            caffe_gpu_gemv(CblasNoTrans, inner_num_, batch_size, Dtype(1.0), sub_mutable, device_scalar, Dtype(0), device_tmp);
            /*
            cublasDgemv(handle,
                batch_size, inner_num_,
                &Dtype(1.0),
                sub_mutable, batch_size,
                device_scalar, 1,
                &Dtype(0),
                mat[i], 1
            )*/
            cudaMemcpy(mat[i], device_tmp, batch_size*sizeof(Dtype), cudaMemcpyDeviceToHost);
            Dtype* val = mat[i];
            for (int j = 0; j < batch_size; j++){
                if (j != i && labels[j] == label){ // j is the positive
                    for (int k = 0; k < batch_size; k++){
                        if (labels[k] != label) {
                            if (val[j]+alpha_ >= val[k]) { // k is the negative
                                loss += val[j] + alpha_ - val[k];
                                // store half of the gradients
                                caffe_gpu_sub(inner_num_, bottom_data+k*inner_num_, bottom_data+j*inner_num_, diff_diff+i*inner_num_);
                            }
                        }
                    }
                }
            }
        }

    }


    top[0]->mutable_cpu_data()[0] = loss;
    for (int i = 0; i < batch_size; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        if (top[0]->cpu_diff()[0] != Dtype(1.0)){
            LOG(INFO) << "Triplet.cu top cpu_diff is not 1.0 is " << top[0]->cpu_diff()[0];
        }

        Dtype scale = Dtype(2.0);
        caffe_gpu_scale(
                bottom[0]->count(),    // count
                scale,                 // scale
                diff_.gpu_diff(),      // input
                bottom[0]->mutable_gpu_diff() // output
        );
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT TripletLossLayer::Backward_cpu" << std::endl;
    }

  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     const Dtype sign = (i == 0) ? 1 : -1;
  //     const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  //     caffe_gpu_axpby(
  //         bottom[i]->count(),              // count
  //         alpha,                              // alpha
  //         diff_.gpu_data(),                   // a
  //         Dtype(0),                           // beta
  //         bottom[i]->mutable_gpu_diff());  // b
  //   }
  // }
}

INSTANTIATE_LAYER_GPU_FUNCS(TripletLossLayer);
} // namespace caffe
