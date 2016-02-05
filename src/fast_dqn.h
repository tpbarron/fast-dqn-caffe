#ifndef SRC_FAST_DQN_H_
#define SRC_FAST_DQN_H_

#include "environment.h"
#include <caffe/caffe.hpp>
#include <memory>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>
#include <deque>
#include <string>
// #include <caffe/layers/memory_data_layer.hpp>

namespace fast_dqn {

constexpr auto kCroppedVolumeSize = 20;
constexpr auto kCroppedVolumeDataSize = kCroppedVolumeSize * kCroppedVolumeSize * kCroppedVolumeSize;
constexpr auto kInputVolumeCount = 4;
constexpr auto kInputDataSize = kCroppedVolumeDataSize * kInputVolumeCount;
constexpr auto kMinibatchSize = 32;
constexpr auto kMinibatchDataSize = kInputDataSize * kMinibatchSize;
constexpr auto kGamma = 0.95f;
constexpr auto kOutputCount = 7;

constexpr auto volumes_layer_name = "volumes_input_layer";
constexpr auto target_layer_name = "target_input_layer";
constexpr auto filter_layer_name = "filter_input_layer";

constexpr auto train_volumes_blob_name = "volumes";
constexpr auto test_volumes_blob_name  = "all_volumes";
constexpr auto target_blob_name       = "target";
constexpr auto filter_blob_name       = "filter";
constexpr auto q_values_blob_name     = "q_values";

using VolumeData = std::array<uint8_t, kCroppedVolumeDataSize>;
using VolumeDataSp = std::shared_ptr<VolumeData>;
using State = std::array<VolumeDataSp, kInputVolumeCount>;
using InputStateBatch = std::vector<State>;


using VolumeLayerInputData = std::array<float, kMinibatchDataSize>;
using TargetLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;
using FilterLayerInputData = std::array<float, kMinibatchSize * kOutputCount>;


typedef struct ActionValue {
  ActionValue(const Environment::ActionCode _action, const float _q_value) : 
    action(_action), q_value(_q_value) {
    }
  const Environment::ActionCode action;
  const float q_value;
} ActionValue;


/**
  * Transition
  */
class Transition {
 public:

  Transition ( const State state, Environment::ActionCode action,
                double reward, VolumeDataSp next_volume ) :
      state_ ( state ),
      action_ ( action ),
      reward_ ( reward ),
      next_volume_ ( next_volume ) {
  }

  bool is_terminal() const { return next_volume_ == nullptr; } 
  
  const State GetNextState() const;
  
  const State& GetState() const { return state_; }
  
  Environment::ActionCode GetAction() const { return action_; }
  
  double GetReward() const { return reward_; }

 private:
    const State state_;
    Environment::ActionCode action_;
    double reward_;
    VolumeDataSp next_volume_;
};
typedef std::shared_ptr<Transition> TransitionSp;

/**
 * Deep Q-Network
 */
class Fast_DQN {
 public:
  Fast_DQN(
      EnvironmentSp environmentSp,
      const Environment::ActionVec& legal_actions,
      const std::string& solver_param,
      const int replay_memory_capacity,
      const double gamma,
      const bool verbose) :
        environmentSp_(environmentSp),
        legal_actions_(legal_actions),
        solver_param_(solver_param),
        replay_memory_capacity_(replay_memory_capacity),
        gamma_(gamma),
        verbose_(verbose),
        random_engine_(0), 
        clone_frequency_(10000), // How often (steps) the target_net_ is updated
        last_clone_iter_(0) {   // Iteration in which the net was last cloned
        }

  /**
   * Initialize DQN. Must be called before calling any other method.
   */
  void Initialize();

  /**
   * Load a trained model from a file.
   */
  void LoadTrainedModel(const std::string& model_file);

  /**
   * Select an action by epsilon-greedy.
   */
  Environment::ActionCode SelectAction(const State& input_frames, double epsilon);

  /**
   * Add a transition to replay memory
   */
  void AddTransition(const Transition& transition);

  /**
   * Update DQN using one minibatch
   */
  void Update();

  int memory_size() const { return replay_memory_.size(); }

  /**
   * Copy the current training net_ to the target_net_
   */
    void CloneTrainingNetToTargetNet() { CloneNet(net_); }

  /**
   * Return the current iteration of the solver
   */
  int current_iteration() const { return solver_->iter(); }

 private:
  using SolverSp = std::shared_ptr<caffe::Solver<float>>;
  using NetSp = boost::shared_ptr<caffe::Net<float>>;
  using BlobSp = boost::shared_ptr<caffe::Blob<float>>;
  using MemoryDataLayerSp = boost::shared_ptr<caffe::MemoryDataLayer<float>>;


  Environment::ActionVec SelectActions(const InputStateBatch& volumes_batch,
                              const double epsilon);
  ActionValue SelectActionGreedily(NetSp net,
                                   const State& last_volumes);
  std::vector<ActionValue> SelectActionGreedily(NetSp,
                                   const InputStateBatch& last_volumes);

  /**
    * Clone the given net and store the result in clone_net_
    */
  void CloneNet(NetSp net);
  
  /**
   * Init the target and filter layers.
   */
  void InitNet(NetSp net);

  /**
    * Input data into the Frames/Target/Filter layers of the given
    * net. This must be done before forward is called.
    */
  void InputDataIntoLayers(NetSp net,
      const VolumeLayerInputData& volume_data,
      const TargetLayerInputData& target_data,
      const FilterLayerInputData& filter_data);

  void PrintNetwork();

  EnvironmentSp environmentSp_;
  const Environment::ActionVec legal_actions_;
  const int replay_memory_capacity_;
  const double gamma_;
  std::deque<Transition> replay_memory_;
  TargetLayerInputData dummy_input_data_;

  const std::string solver_param_;
  SolverSp solver_;
  NetSp net_; // The primary network used for action selection.
  NetSp target_net_; // Clone used to generate targets.
  const int clone_frequency_; // How often (steps) the target_net is updated
  int last_clone_iter_; // Iteration in which the net was last cloned

  
  std::mt19937 random_engine_;
  bool verbose_;
};


}  // namespace fast_dqn

#endif  // SRC_FAST_DQN_H_
