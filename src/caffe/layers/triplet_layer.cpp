#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
namespace caffe {

/// find triplet(X_i^a, X_i^p, X_i^n) satisfies some constraint
/// X_i^p is the positive, means has the same label as X_i^a
/// X_i^n is the negative, means has the different labels as X_i^a

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // top is just a scalar
  LossLayer<Dtype>::Reshape(bottom, top);
  // store (X_i^n - X_i^p)/N which can backpropagate to prev-layer directly
  diff_.ReshapeLike(*bottom[0]); // bottom[0] is batch_size*channels(128)*1*1
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype loss = Dtype(0);
    // it is the hardest part which try to find the triplet
    // algorithm should be fast enough and we just choose semi-hard triplet
    // refer to the FaceNet paper for more detail
    int num = bottom[0]->num(); // get the batch_size
    int examples = num;
    // examples should be less than num
    for (int i = 0; i < examples; i++){
        
    }

    top[0]->mutable_cpu_data()[0] = loss;

  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        Dtype scale = Dtype(-1.0)/bottom[0]->num();
        caffe_cpu_scale(
                bottom[0]->count(),    // count
                scale,                 // scale
                diff_.cpu_data(),      // input
                bottom[0]->mutable_cpu_diff() // output
        );
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT TripletLossLayer::Backward_cpu" << endl;
    }

  // for (int i = 0; i < 2; ++i) {
  //   if (propagate_down[i]) {
  //     const Dtype sign = (i == 0) ? 1 : -1;
  //     const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
  //     caffe_cpu_axpby(
  //         bottom[i]->count(),              // count
  //         alpha,                              // alpha
  //         diff_.cpu_data(),                   // a
  //         Dtype(0),                           // beta
  //         bottom[i]->mutable_cpu_diff());  // b
  //   }
  // }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);
} // namespace caffe
