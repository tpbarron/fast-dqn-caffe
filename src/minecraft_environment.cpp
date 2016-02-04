#include "environment.h"
#include <minecraft_dqn_interface.hpp>
#include <glog/logging.h>
#include <iostream>
#include <vector>

namespace fast_dqn {

class MinecraftEnvironment : public Environment {

public:

  MinecraftEnvironment(int argc, char *argv[], const std::string path, bool evaluate) :
      me_(argc, argv, path) {
    // init the python game
    me_.init(evaluate);
    
    ActionVec av = me_.get_action_set();
    for (int i=0; i < av.size(); i++) {
      legal_actions_.push_back(av[i]);
      std::cout << "action[" << i << "] = " << av[i] << std::endl;
    }
  }


  /**
   * Preprocess screen does very little with the minecraft environment
   * It literally copies the grayscale pixels from the cv::Mat image into
   * a pointer array of bytes. 
   * TODO: this could be made more efficient by directly passing a 1d array from 
   * the minecraft game
   */
  VolumeDataSp PreprocessScreen() {
    std::vector<uint8_t> raw_volume = me_.get_volume();

    //assert(raw_screen.cols == kCroppedFrameSize);
    //assert(raw_screen.rows == kCroppedFrameSize);
  
    auto volume = std::make_shared<VolumeData>();
    for (auto i = 0; i < (int)std::pow(kCroppedVolumeSize, 3.0); ++i) {
      (*volume)[i] = raw_volume[i]; //resulting_color;
    }
    
    return volume;
  }

  /**
   * TODO: define the actions in a way that they can be accessed by both c++ and 
   * Python
   */
  double ActNoop() {
    double reward = 0;
    for (auto i = 0; i < kInputVolumeCount && !me_.is_game_over(); ++i) {
      //std::cout << "actnoop, i = " << i << std::endl;
      // TODO: Should this be an int or some type of action object?
      // Temporarility 0 will always be No-op
      reward += me_.act(0);
    }
    return reward;
  }

  double Act(int action) {
    double reward = 0;
    for (auto i = 0; i < kInputVolumeCount && !me_.is_game_over(); ++i) {
      // TODO: action type?
      reward += me_.act(action);
    }
    //std::cout << "Minecraft env reward: " << reward << std::endl;
    return reward;
  }

  void Reset() { 
    me_.reset(); 
  }

  bool EpisodeOver() { 
    return me_.is_game_over(); 
  }

  std::string action_to_string(Environment::ActionCode a) {
    return "print action: " + std::to_string(a);
    //return action_to_string(static_cast<Action>(a)); 
  }

  const ActionVec& GetMinimalActionSet() {
    return legal_actions_;
  }

 private:

  MinecraftInterface me_;
  ActionVec legal_actions_;
  
};

EnvironmentSp CreateEnvironment(int argc,
				char *argv[],
				const std::string path,
				bool evaluate) {
  return std::make_shared<MinecraftEnvironment>(argc, argv, path, evaluate);
}

}  // namespace fast_dqn
