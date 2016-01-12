#include "environment.h"
#include <minecraft_dqn_interface.hpp>
#include <glog/logging.h>
#include <iostream>
#include <vector>

namespace fast_dqn {

class MinecraftEnvironment : public Environment {

public:

  MinecraftEnvironment(int argc, char *argv[], const std::string path) :
      me_(argc, argv, path) {
    // init the python game
    me_.init();
    
    ActionVec av = me_.get_action_set();
    for (int i=0; i < av.size(); i++) {
      legal_actions_.push_back(av[i]);
      //std::cout << "action " << i << " = " << av[i] << std::endl;
    }
  }


  /**
   * Preprocess screen does very little with the minecraft environment
   * It literally copies the grayscale pixels from the cv::Mat image into
   * a pointer array of bytes. 
   * TODO: this could be made more efficient by directly passing a 1d array from 
   * the minecraft game
   */
  FrameDataSp PreprocessScreen() {
    /**
     * Uncomment the foloowing line and comment the remainder of the method
     * to get the screen as an array from the minecraft interface. For some
     * reason this causes an error on the action on the following step.
     */
    // return me_.get_screen_as_array();
    
    cv::Mat raw_screen = me_.get_screen();
    assert(raw_screen.cols == kCroppedFrameSize);
    assert(raw_screen.rows == kCroppedFrameSize);
  
    auto screen = std::make_shared<FrameData>();
    for (auto i = 0; i < kCroppedFrameSize; ++i) {
      for (auto j = 0; j < kCroppedFrameSize; ++j) {
        (*screen)[i * kCroppedFrameSize + j] = raw_screen.at<uchar>(i, j); //resulting_color;
      }
    }
    
    return screen;
  }

  /**
   * TODO: define the actions in a way that they can be accessed by both c++ and 
   * Python
   */
  double ActNoop() {
    double reward = 0;
    for (auto i = 0; i < kInputFrameCount && !me_.is_game_over(); ++i) {
      //std::cout << "actnoop, i = " << i << std::endl;
      // TODO: Should this be an int or some type of action object?
      // Temporarility 0 will always be No-op
      reward += me_.act(0);
    }
    return reward;
  }

  double Act(int action) {
    double reward = 0;
    for (auto i = 0; i < kInputFrameCount && !me_.is_game_over(); ++i) {
      // TODO: action type?
      reward += me_.act(action);
    }
    //std::cout << "Reward: " << reward << std::endl;
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

EnvironmentSp CreateEnvironment(int argc, char *argv[], const std::string path) {
  return std::make_shared<MinecraftEnvironment>(argc, argv, path);
}

}  // namespace fast_dqn
