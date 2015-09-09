// add by Binbin Xu
// declanxu@gmail.com or declanxu@126.com
// Zhejiang University, State Key Lab of CAD&CG.

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
namespace caffe {

/// find triplet(X_i^a, X_i^p, X_i^n) satisfies some constraint
/// X_i^p is the positive, means has the same label as X_i^a
/// X_i^n is the negative, means has the different labels as X_i^a

// in layer.hpp Reshape will be called after LayerSetUp
template <typename Dtype>
void OffTripletLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
    LossLayer<Dtype>::Reshape(bottom, top);
    diff_.ReshapeLike(*bottom[0]); // bottom[0] is batch_size*channels(128)*1*1
    inner_num_ = bottom[0]->count(1);
}

template <typename Dtype>
void OffTripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // top is just a scalar
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // store (X_i^n - X_i^p)/N which can backpropagate to prev-layer directly
  // diff_.ReshapeLike(*bottom[0]); // bottom[0] is batch_size*channels(128)*1*1
  // sub_.ReshapeLike(*bottom[0]);
  inner_num_ = bottom[0]->count(1);

  // get some parameters
  // label_separator_ = this->layer_param_.triplet_loss_param().separate();
  // identities_per_batch_ = this->layer_param_.triplet_loss_param().ids_per_batch();
  // num_per_identity_ = this->layer_param_.triplet_loss_param().num_per_id();
  alpha_ = (Dtype)(this->layer_param_.off_triplet_loss_param().alpha());

}

template <typename Dtype>
void OffTripletLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    // it is the hardest part which is trying to find the triplet
    // algorithm should be fast enough and we just choose semi-hard triplets
    // refer to the paper: FaceNet for more details
    Dtype loss = Dtype(0);
    int batch_size = bottom[0]->num(); // get the batch_size
    //CHECK_EQ(batch_size, bottom[1]->count());
    CHECK_EQ(inner_num_, 128);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* sub = diff_.mutable_cpu_data();
    Dtype* diff = diff_.mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0.0), diff);

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
      caffe_sub(inner_num_, bottom_data + i*inner_num_, bottom_data + label_pos[i], sub);
      res += caffe_cpu_dot(inner_num_, sub, sub);
      caffe_sub(inner_num_, bottom_data + i*inner_num_, bottom_data + label_neg[i], sub);
      res -= caffe_cpu_dot(inner_num_, sub, sub);
      if (res > 0.0) {
        loss += res;
        caffe_sub(inner_num_, bottom_data + label_neg[i], bottom_data + label_pos[i], diff + i*inner_num_);
      }
    }
    // NOT_IMPLEMENTED;
    top[0]->mutable_cpu_data()[0] = loss;

}

template <typename Dtype>
void OffTripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[0]){
        //Dtype scale = Dtype(2.0);
        Dtype scale = Dtype(2.0)*top[0]->cpu_diff()[0]/bottom[0]->num();
        caffe_cpu_scale(
                bottom[0]->count(),    // count
                scale,                 // scale
                diff_.cpu_diff(),      // input
                bottom[0]->mutable_cpu_diff() // output
        );
    }
    else {
        LOG(ERROR) << "should be back propagate to prev-layer AT OffTripletLossLayer::Backward_cpu" << std::endl;
    }
}

#ifdef CPU_ONLY
STUB_GPU(OffTripletLossLayer);
#endif

INSTANTIATE_CLASS(OffTripletLossLayer);
REGISTER_LAYER_CLASS(OffTripletLoss);
} // namespace caffe
