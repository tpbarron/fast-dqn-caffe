#include "fast_dqn.h"
#include "environment.h"
#include <glog/logging.h>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <sstream>
#include <utility>
#include <string>
#include <vector>

namespace fast_dqn {


std::string PrintQValues(
    EnvironmentSp environmentSp,
    const std::vector<float>& q_values, const Environment::ActionVec& actions) {
  assert(!q_values.empty());
  assert(!actions.empty());
  assert(q_values.size() == actions.size());
  std::ostringstream actions_buf;
  std::ostringstream q_values_buf;
  for (auto i = 0; i < q_values.size(); ++i) {
    const auto a_str =
        boost::algorithm::replace_all_copy(
            environmentSp->action_to_string(actions[i]), "PLAYER_A_", "");
    const auto q_str = std::to_string(q_values[i]);
    const auto column_size = std::max(a_str.size(), q_str.size()) + 1;
    actions_buf.width(column_size);
    actions_buf << a_str;
    q_values_buf.width(column_size);
    q_values_buf << q_str;
  }
  actions_buf << std::endl;
  q_values_buf << std::endl;
  return actions_buf.str() + q_values_buf.str();
}


const State Transition::GetNextState() const {

  //  Create the s(t+1) states from the experience(t)'s

  if (next_frame_ == nullptr) {
    // Terminal state so no next_observation, just return current state
    return state_;
  } else {
    State state_clone;

    for (int i = 0; i < kInputFrameCount - 1; ++i)
      state_clone[i] = state_[i + 1];
    state_clone[kInputFrameCount - 1] = next_frame_;
    return state_clone;
  }

}

template <typename Dtype>
void HasBlobSize(caffe::Net<Dtype>& net,
                 const std::string& blob_name,
                 const std::vector<int> expected_shape) {
  net.has_blob(blob_name);
  const caffe::Blob<Dtype>& blob = *net.blob_by_name(blob_name);
  const std::vector<int>& blob_shape = blob.shape();
  CHECK_EQ(blob_shape.size(), expected_shape.size());
  CHECK(std::equal(blob_shape.begin(), blob_shape.end(),
                   expected_shape.begin()));
}

void Fast_DQN::LoadTrainedModel(const std::string& model_bin) {
  net_->CopyTrainedLayersFrom(model_bin);
  CloneTrainingNetToTargetNet();
}


void PopulateLayer(caffe::LayerParameter& layer,
                   const std::string& name, const std::string& type,
                   const std::vector<std::string>& bottoms,
                   const std::vector<std::string>& tops,
                   const boost::optional<caffe::Phase>& include_phase) {
  layer.set_name(name);
  layer.set_type(type);
  for (auto& bottom : bottoms) {
    layer.add_bottom(bottom);
  }
  for (auto& top : tops) {
    layer.add_top(top);
  }
  // PopulateLayer(layer, name, type, bottoms, tops);
  if (include_phase) {
    layer.add_include()->set_phase(*include_phase);
  }
}

void MemoryDataLayer(caffe::NetParameter& net_param,
                     const std::string& name,
                     const std::vector<std::string>& tops,
                     const boost::optional<caffe::Phase>& include_phase,
                     const std::vector<int>& shape) {
  caffe::LayerParameter& memory_layer = *net_param.add_layer();
  PopulateLayer(memory_layer, name, "MemoryData", {}, tops, include_phase);
  CHECK_EQ(shape.size(), 4);
  caffe::MemoryDataParameter* memory_data_param =
      memory_layer.mutable_memory_data_param();
  memory_data_param->set_batch_size(shape[0]);
  memory_data_param->set_channels(shape[1]);
  memory_data_param->set_height(shape[2]);
  memory_data_param->set_width(shape[3]);
}

void ReshapeLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const std::vector<int>& shape) {
  caffe::LayerParameter& reshape_layer = *net_param.add_layer();
  PopulateLayer(reshape_layer, name, "Reshape", bottoms, tops, include_phase);
  caffe::ReshapeParameter* reshape_param = reshape_layer.mutable_reshape_param();
  caffe::BlobShape* blob_shape = reshape_param->mutable_shape();
  for (auto& dim : shape) {
    blob_shape->add_dim(dim);
  }
}

void SliceLayer(caffe::NetParameter& net_param,
                const std::string& name,
                const std::vector<std::string>& bottoms,
                const std::vector<std::string>& tops,
                const boost::optional<caffe::Phase>& include_phase,
                const int axis,
                const std::vector<int>& slice_points) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Slice", bottoms, tops, include_phase);
  caffe::SliceParameter* slice_param = layer.mutable_slice_param();
  slice_param->set_axis(axis);
  for (auto& p : slice_points) {
    slice_param->add_slice_point(p);
  }
}

void ConvLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const std::string& shared_name,
               const float& lr_mult,
               const boost::optional<caffe::Phase>& include_phase,
               const int num_output,
               const int kernel_size,
               const int stride) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Convolution", bottoms, tops, include_phase);
  caffe::ParamSpec* weight_param = layer.add_param();
  weight_param->set_name(shared_name + "_w");
  if (lr_mult >= 0) {
    weight_param->set_lr_mult(lr_mult);
  }
  weight_param->set_decay_mult(1);
  caffe::ParamSpec* bias_param = layer.add_param();
  bias_param->set_name(shared_name + "_b");
  if (lr_mult >= 0) {
    bias_param->set_lr_mult(2 * lr_mult);
  }
  bias_param->set_decay_mult(0);
  caffe::ConvolutionParameter* conv_param = layer.mutable_convolution_param();
  conv_param->set_num_output(num_output);
  conv_param->set_kernel_size(kernel_size);
  conv_param->set_stride(stride);
  caffe::FillerParameter* weight_filler = conv_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.01);
  caffe::FillerParameter* bias_filler = conv_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(0);
}

void ReluLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "ReLU", bottoms, tops, include_phase);
  caffe::ReLUParameter* relu_param = layer.mutable_relu_param();
  relu_param->set_negative_slope(0.01);
}

void IPLayer(caffe::NetParameter& net_param,
             const std::string& name,
             const std::vector<std::string>& bottoms,
             const std::vector<std::string>& tops,
             const std::string& shared_name,
             const float& lr_mult,
             const boost::optional<caffe::Phase>& include_phase,
             const int num_output,
             const int axis) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "InnerProduct", bottoms, tops, include_phase);
  caffe::ParamSpec* weight_param = layer.add_param();
  weight_param->set_name(shared_name + "_w");
  if (lr_mult >= 0) {
    weight_param->set_lr_mult(lr_mult);
  }
  weight_param->set_decay_mult(1);
  caffe::ParamSpec* bias_param = layer.add_param();
  bias_param->set_name(shared_name + "_b");
  if (lr_mult >= 0) {
    bias_param->set_lr_mult(2 * lr_mult);
  }
  bias_param->set_decay_mult(0);
  caffe::InnerProductParameter* ip_param = layer.mutable_inner_product_param();
  ip_param->set_num_output(num_output);
  ip_param->set_axis(axis);
  caffe::FillerParameter* weight_filler = ip_param->mutable_weight_filler();
  weight_filler->set_type("gaussian");
  weight_filler->set_std(0.005);
  caffe::FillerParameter* bias_filler = ip_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(1);
}

void ConcatLayer(caffe::NetParameter& net_param,
                 const std::string& name,
                 const std::vector<std::string>& bottoms,
                 const std::vector<std::string>& tops,
                 const boost::optional<caffe::Phase>& include_phase,
                 const int& axis) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Concat", bottoms, tops, include_phase);
  caffe::ConcatParameter* concat_param = layer.mutable_concat_param();
  concat_param->set_axis(axis);
}

void LstmLayer(caffe::NetParameter& net_param,
               const std::string& name,
               const std::vector<std::string>& bottoms,
               const std::vector<std::string>& tops,
               const boost::optional<caffe::Phase>& include_phase,
               const int& num_output) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "LSTM", bottoms, tops, include_phase);
  caffe::RecurrentParameter* recurrent_param = layer.mutable_recurrent_param();
  recurrent_param->set_num_output(num_output);
  caffe::FillerParameter* weight_filler = recurrent_param->mutable_weight_filler();
  weight_filler->set_type("uniform");
  weight_filler->set_min(-0.08);
  weight_filler->set_max(0.08);
  caffe::FillerParameter* bias_filler = recurrent_param->mutable_bias_filler();
  bias_filler->set_type("constant");
  bias_filler->set_value(0);
}

void EltwiseLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase,
                  const caffe::EltwiseParameter::EltwiseOp& op) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Eltwise", bottoms, tops, include_phase);
  caffe::EltwiseParameter* eltwise_param = layer.mutable_eltwise_param();
  eltwise_param->set_operation(op);
}

void SilenceLayer(caffe::NetParameter& net_param,
                  const std::string& name,
                  const std::vector<std::string>& bottoms,
                  const std::vector<std::string>& tops,
                  const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "Silence", bottoms, tops, include_phase);
}

void EuclideanLossLayer(caffe::NetParameter& net_param,
                        const std::string& name,
                        const std::vector<std::string>& bottoms,
                        const std::vector<std::string>& tops,
                        const boost::optional<caffe::Phase>& include_phase) {
  caffe::LayerParameter& layer = *net_param.add_layer();
  PopulateLayer(layer, name, "EuclideanLoss", bottoms, tops, include_phase);
}

caffe::NetParameter Fast_DQN::CreateNet(bool unroll1_is_lstm) {
  caffe::NetParameter np;
  np.set_name("Deep Recurrent Q-Network");
  MemoryDataLayer(
      np, frames_layer_name, {train_frames_blob_name,"dummy_frames"}, caffe::TRAIN,
      {kMinibatchSize, frames_per_forward_, kCroppedFrameSize, kCroppedFrameSize});
  MemoryDataLayer(
      np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TRAIN,
      {unroll_, kMinibatchSize, 1, 1});
  MemoryDataLayer(
      np, target_layer_name, {target_blob_name,"dummy_target"}, caffe::TRAIN,
      {unroll_, kMinibatchSize, kOutputCount, 1});
  MemoryDataLayer(
      np, filter_layer_name, {filter_blob_name,"dummy_filter"}, caffe::TRAIN,
      {unroll_, kMinibatchSize, kOutputCount, 1});
  SilenceLayer(np, "silence", {"dummy_frames","dummy_cont","dummy_filter",
          "dummy_target"}, {}, caffe::TRAIN);
  ReshapeLayer(
      np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TRAIN,
      {unroll_, kMinibatchSize});
  ReshapeLayer(
      np, "reshape_filter", {filter_blob_name}, {"reshaped_filter"}, caffe::TRAIN,
      {unroll_, kMinibatchSize, kOutputCount});
  MemoryDataLayer(
      np, frames_layer_name, {test_frames_blob_name,"dummy_frames"},
      caffe::TEST,
      {kMinibatchSize,kInputFrameCount,kCroppedFrameSize,kCroppedFrameSize});
  MemoryDataLayer(
      np, cont_layer_name, {cont_blob_name,"dummy_cont"}, caffe::TEST,
      {1, kMinibatchSize, 1, 1});
  SilenceLayer(np, "silence", {"dummy_frames","dummy_cont"}, {}, caffe::TEST);
  ReshapeLayer(
      np, "reshape_cont", {cont_blob_name}, {"reshaped_cont"}, caffe::TEST,
      {1, kMinibatchSize});
  if (unroll_ > 1) {
    std::vector<std::string> frames_tops, scrap_tops;
    for (int t = 0; t < unroll_; ++t) {
      std::string ts = std::to_string(t);
      boost::optional<caffe::Phase> phase;
      if (t > 0) { phase.reset(caffe::TRAIN); }
      std::vector<int> slice_points;
      std::vector<std::string> slice_tops;
      if (t == 0) {
        slice_points = {kInputFrameCount};
        slice_tops = {"frames_"+ts, "scrap_"+ts};
        scrap_tops.push_back("scrap_"+ts);
      } else if (t == unroll_ - 1) {
        slice_points = {t};
        slice_tops = {"scrap_"+ts, "frames_"+ts};
        scrap_tops.push_back("scrap_"+ts);
      } else {
        slice_tops = {"scrap1_"+ts, "frames_"+ts, "scrap2_"+ts};
        scrap_tops.push_back("scrap1_"+ts);
        scrap_tops.push_back("scrap2_"+ts);
        slice_points = {t, t + kInputFrameCount};
      }
      SliceLayer(np, "slice_"+ts, {train_frames_blob_name}, slice_tops,
                 caffe::TRAIN, 1, slice_points);
      frames_tops.push_back("frames_"+ts);
    }
    SilenceLayer(np, "scrap_silence", scrap_tops, {}, caffe::TRAIN);
    ConcatLayer(np, "concat_frames", frames_tops, {"all_frames"}, caffe::TRAIN,0);
    ConvLayer(np, "conv1", {"all_frames"}, {"conv1"}, "conv1", -1,
              boost::none, 32, 8, 4);
  } else {
    ConvLayer(np, "conv1", {train_frames_blob_name}, {"conv1"}, "conv1", -1,
              caffe::TRAIN, 32, 8, 4);
    ConvLayer(np, "conv1", {"all_frames"}, {"conv1"}, "conv1", -1, caffe::TEST,
              32, 8, 4);
  }
  ReluLayer(np, "conv1_relu", {"conv1"}, {"conv1"}, boost::none);
  ConvLayer(np, "conv2", {"conv1"}, {"conv2"}, "conv2", -1, boost::none,
            64, 4, 2);
  ReluLayer(np, "conv2_relu", {"conv2"}, {"conv2"}, boost::none);
  ConvLayer(np, "conv3", {"conv2"}, {"conv3"}, "conv3", -1, boost::none,
            64, 3, 1);
  ReluLayer(np, "conv3_relu", {"conv3"}, {"conv3"}, boost::none);
  ReshapeLayer(np, "conv3_reshape", {"conv3"}, {"reshaped_conv3"},
               caffe::TRAIN, {unroll_, kMinibatchSize, 64*7*7});
  ReshapeLayer(np, "conv3_reshape", {"conv3"}, {"reshaped_conv3"},
               caffe::TEST, {1, kMinibatchSize, 64*7*7});
  if (unroll_ > 1 || unroll1_is_lstm) {
    LstmLayer(np, "lstm1", {"reshaped_conv3","reshaped_cont"}, {"lstm1"}, boost::none,
              lstmSize);
  } else {
    IPLayer(np, "lstm1", {"reshaped_conv3"}, {"lstm1"}, "lstm1", -1, boost::none,
            lstmSize, 2);
    ReluLayer(np, "ip1_relu", {"lstm1"}, {"lstm1"}, boost::none);
    SilenceLayer(np, "cont_silence", {"reshaped_cont"}, {}, boost::none);
  }

  IPLayer(np, "ip2", {"lstm1"}, {q_values_blob_name}, "ip2", -1, boost::none,
          kOutputCount, 2);
  EltwiseLayer(np, "eltwise_filter", {q_values_blob_name,"reshaped_filter"},
               {"filtered_q_values"}, caffe::TRAIN, caffe::EltwiseParameter::PROD);
  EuclideanLossLayer(np, "loss", {"filtered_q_values","target"}, {"loss"},
                     caffe::TRAIN);
  return np;
}

void Fast_DQN::Initialize() {

  // Initialize dummy input data with 0
  std::fill(dummy_input_data_.begin(), dummy_input_data_.end(), 0.0);
  
  // Construct the solver
  caffe::SolverParameter solver_param;
  caffe::ReadProtoFromTextFileOrDie(solver_param_, &solver_param);
  caffe::NetParameter* net_param = solver_param.mutable_net_param();
  net_param->CopyFrom(CreateNet(unroll1_is_lstm_));
  std::string net_filename = "models/ale_recurrent_net.prototxt"; //save_path.native() + "_net.prototxt";
  WriteProtoToTextFile(*net_param, net_filename.c_str());
  //solver_param.set_snapshot_prefix("model/");

  // Initialize net and solver
  solver_.reset(caffe::GetSolver<float>(solver_param));

  // New solver creation API.  Caution, caffe master current doesn't
  // work.  Something broke the training.
  // use commit:ff16f6e43dd718921e5203f640dd57c68f01cdb3 for now.  It's slower
  // though.  Let me know if you figure out the issue.
  // solver_.reset(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  net_ = solver_->net();
  //std::cout << "Pre net initalization" << std::endl;
  InitNet(net_);
  //std::cout << "Post net initalization" << std::endl; 
  
  //CHECK_EQ(solver_->test_nets().size(), 1);
  //target_net_ = solver_->test_nets()[0];
  // Test Net shares parameters with train net at all times
  //target_net_->ShareTrainedLayersWith(net_.get());
  // Clone net maintains its own set of parameters
  //CloneNet(target_net_);

  CloneTrainingNetToTargetNet();
  
  // Check the primary network
  HasBlobSize(*net_, train_frames_blob_name, {kMinibatchSize,
          frames_per_forward_, kCroppedFrameSize, kCroppedFrameSize});
  HasBlobSize(*net_, target_blob_name, {unroll_, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*net_, filter_blob_name, {unroll_, kMinibatchSize, kOutputCount, 1});
  HasBlobSize(*net_, cont_blob_name, {unroll_, kMinibatchSize, 1, 1});

  LOG(INFO) << "Finished " << net_->name() << " Initialization";
}


Environment::ActionCode Fast_DQN::SelectAction(const State& frames, 
                                               const double epsilon) {
  return SelectActions(InputStateBatch{{frames}}, epsilon)[0];
}

Environment::ActionVec Fast_DQN::SelectActions(
                              const InputStateBatch& frames_batch,
                              const double epsilon) {
  CHECK(epsilon <= 1.0 && epsilon >= 0.0);
  CHECK_LE(frames_batch.size(), kMinibatchSize);
  Environment::ActionVec actions(frames_batch.size());
  if (std::uniform_real_distribution<>(0.0, 1.0)(random_engine_) < epsilon) {
    // Select randomly
    for (int i = 0; i < actions.size(); ++i) {
      const auto random_idx = std::uniform_int_distribution<int>
          (0, legal_actions_.size() - 1)(random_engine_);
      actions[i] = legal_actions_[random_idx];
    }
  } else {
    // Select greedily
    std::vector<ActionValue> actions_and_values =
        SelectActionGreedily(target_net_, frames_batch);
    CHECK_EQ(actions_and_values.size(), actions.size());
    for (int i=0; i<actions_and_values.size(); ++i) {
      actions[i] = actions_and_values[i].action;
    }
  }
  return actions;
}


ActionValue Fast_DQN::SelectActionGreedily(
  NetSp net,
  const State& last_frames) {
  return SelectActionGreedily(net, InputStateBatch{{last_frames}}).front();
}

std::vector<ActionValue> Fast_DQN::SelectActionGreedily(
    NetSp net,
    const InputStateBatch& last_frames_batch) {
  assert(last_frames_batch.size() <= kMinibatchSize);
  std::array<float, kMinibatchDataSize> frames_input;
  for (auto i = 0; i < last_frames_batch.size(); ++i) {
    // Input frames to the net and compute Q values for each legal actions
    for (auto j = 0; j < kInputFrameCount; ++j) {
      const auto& frame_data = last_frames_batch[i][j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize +
              j * kCroppedFrameDataSize);
    }
  }
  InputDataIntoLayers(net, frames_input, cont_input, dummy_input_data_, dummy_input_data_);
  net->ForwardPrefilled(nullptr);

  std::vector<ActionValue> results;
  results.reserve(last_frames_batch.size());
  CHECK(net->has_blob(q_values_blob_name));
  const auto q_values_blob = net->blob_by_name(q_values_blob_name);
  for (auto i = 0; i < last_frames_batch.size(); ++i) {
    // Get the Q values from the net
    const auto action_evaluator = [&](Environment::ActionCode action) {
      const auto q = q_values_blob->data_at(i, static_cast<int>(action), 0, 0);
      assert(!std::isnan(q));
      return q;
    };
    std::vector<float> q_values(legal_actions_.size());
    std::transform(
        legal_actions_.begin(),
        legal_actions_.end(),
        q_values.begin(),
        action_evaluator);
//     if (last_frames_batch.size() == 1) {
//       std::cout << PrintQValues(q_values, legal_actions_);
//     }

    // Select the action with the maximum Q value
    const auto max_idx =
        std::distance(
            q_values.begin(),
            std::max_element(q_values.begin(), q_values.end()));
    results.emplace_back(legal_actions_[max_idx], q_values[max_idx]);
  }
  return results;
}

void Fast_DQN::AddTransition(const Transition& transition) {
  if (replay_memory_.size() == replay_memory_capacity_) {
    replay_memory_.pop_front();
  }
  replay_memory_.push_back(transition);
}

void Fast_DQN::Update() {
  if (verbose_)
    LOG(INFO) << "iteration: " << current_iteration() << std::endl;

  // Every clone_iters steps, update the clone_net_
  if (current_iteration() >= last_clone_iter_ + clone_frequency_) {
    LOG(INFO) << "Iter " << current_iteration() << ": Updating Clone Net";
    CloneTrainingNetToTargetNet();
    last_clone_iter_ = current_iteration();
  }

  // Sample transitions from replay memory
  std::vector<int> transitions;
  transitions.reserve(kMinibatchSize);
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto random_transition_idx =
        std::uniform_int_distribution<int>(0, replay_memory_.size() - 1)(
            random_engine_);
    transitions.push_back(random_transition_idx);
  }

  // Compute target values: max_a Q(s',a)
  std::vector<State> target_last_frames_batch;
  for (const auto idx : transitions) {
    const auto& transition = replay_memory_[idx];
    if (transition.is_terminal()) {
      continue;
    }

    target_last_frames_batch.push_back(transition.GetNextState());
  }

    // Get the next state QValues
  const auto actions_and_values =
      SelectActionGreedily(target_net_, target_last_frames_batch);

  FramesLayerInputData frames_input;
  TargetLayerInputData target_input;
  FilterLayerInputData filter_input;
  std::fill(target_input.begin(), target_input.end(), 0.0f);
  std::fill(filter_input.begin(), filter_input.end(), 0.0f);
  auto target_value_idx = 0;
  for (auto i = 0; i < kMinibatchSize; ++i) {
    const auto& transition = replay_memory_[transitions[i]];
    const auto action = transition.GetAction();
    const auto reward = transition.GetReward();
    assert(reward >= -1.0 && reward <= 1.0);
    const auto target = transition.is_terminal() ?
          reward :
          reward + gamma_ * actions_and_values[target_value_idx++].q_value;
    assert(!std::isnan(target));
    target_input[i * kOutputCount + static_cast<int>(action)] = target;
    filter_input[i * kOutputCount + static_cast<int>(action)] = 1;
    if (verbose_)
      VLOG(1) << "filter:" << environmentSp_->action_to_string(action) 
        << " target:" << target;
    for (auto j = 0; j < kInputFrameCount; ++j) {
      const State& state = transition.GetState();
      const auto& frame_data = state[j];
      std::copy(
          frame_data->begin(),
          frame_data->end(),
          frames_input.begin() + i * kInputDataSize +
              j * kCroppedFrameDataSize);
    }
  }
  InputDataIntoLayers(net_, frames_input, cont_input, target_input, filter_input);
  solver_->Step(1);
  // Log the first parameter of each hidden layer
//   VLOG(1) << "conv1:" <<
//     net_->layer_by_name("conv1_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "conv2:" <<
//     net_->layer_by_name("conv2_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "ip1:" <<
//     net_->layer_by_name("ip1_layer")->blobs().front()->data_at(1, 0, 0, 0);
//   VLOG(1) << "ip2:" <<
//     net_->layer_by_name("ip2_layer")->blobs().front()->data_at(1, 0, 0, 0);
}

void Fast_DQN::InitNet(NetSp net) {
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(target_layer_name));
    CHECK(target_input_layer);
    target_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
                              const_cast<float*>(dummy_input_data_.data()),
                              target_input_layer->batch_size());
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(filter_layer_name));
    CHECK(filter_input_layer);
    filter_input_layer->Reset(const_cast<float*>(dummy_input_data_.data()),
                              const_cast<float*>(dummy_input_data_.data()),
                              filter_input_layer->batch_size());
}

void Fast_DQN::CloneNet(NetSp net) {
  caffe::NetParameter net_param;
  net->ToProto(&net_param);
  net_param.mutable_state()->set_phase(net->phase());
  if (target_net_ == nullptr) {
    target_net_.reset(new caffe::Net<float>(net_param));
  } else {
    target_net_->CopyTrainedLayersFrom(net_param);
  }
  InitNet(target_net_);
}


void Fast_DQN::InputDataIntoLayers(NetSp net,
      const FramesLayerInputData& frames_input,
      const ContLayerInputData&   cont_input,
      const TargetLayerInputData& target_input,
      const FilterLayerInputData& filter_input) {

  const auto frames_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net->layer_by_name(frames_layer_name));
  CHECK(frames_input_layer);

  frames_input_layer->Reset(const_cast<float*>(frames_input.data()),
                            const_cast<float*>(frames_input.data()),
                            frames_input_layer->batch_size());

  const auto cont_input_layer =
      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
          net->layer_by_name(cont_layer_name));
  CHECK(cont_input_layer);
  cont_input_layer->Reset(const_cast<float*>(cont_input.data()), 
                          const_cast<float*>(cont_input.data()),
                          cont_input_layer->batch_size());
                          
  if (net == net_) { // training net?
    const auto target_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(target_layer_name));
    CHECK(target_input_layer);
    target_input_layer->Reset(const_cast<float*>(target_input.data()),
                              const_cast<float*>(target_input.data()),
                              target_input_layer->batch_size());
    const auto filter_input_layer =
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(
            net->layer_by_name(filter_layer_name));
    CHECK(filter_input_layer);
    filter_input_layer->Reset(const_cast<float*>(filter_input.data()),
                              const_cast<float*>(filter_input.data()),
                              filter_input_layer->batch_size());
  }

}

}  // namespace fast_dqn

