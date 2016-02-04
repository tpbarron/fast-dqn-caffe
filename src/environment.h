#ifndef SRC_ENVIRONMENT_H_
#define SRC_ENVIRONMENT_H_
#include <vector>
#include <memory>

namespace fast_dqn {

  // Abstract environment class
  // implementation must define the class 
  //    EnvironmentSp CreateEnvironment( bool gui, const std::string rom_path);

class Environment;
typedef std::shared_ptr<Environment> EnvironmentSp;

class Environment {
 public:
  typedef std::vector<int> ActionVec;
  typedef int ActionCode;

  static constexpr auto kCroppedVolumeSize = 100;
  static constexpr auto kCroppedVolumeDataSize = 
    kCroppedVolumeSize * kCroppedVolumeSize * kCroppedVolumeSize;
  static constexpr auto kInputVolumeCount = 4;
  static constexpr auto kInputDataSize = 
    kCroppedVolumeDataSize * kInputVolumeCount;

  using VolumeData = std::array<uint8_t, kCroppedVolumeDataSize>;
  using VolumeDataSp = std::shared_ptr<VolumeData>;
  using State = std::array<VolumeDataSp, kInputVolumeCount>;

  virtual VolumeDataSp PreprocessScreen() = 0;

  virtual double ActNoop() = 0;

  virtual double Act(int action) = 0;

  virtual void Reset() = 0;

  virtual bool EpisodeOver() = 0;

  virtual std::string action_to_string(ActionCode a) = 0;

  virtual const ActionVec& GetMinimalActionSet() = 0;

};

// Factory method
EnvironmentSp CreateEnvironment(bool gui, const std::string rom_path);
EnvironmentSp CreateEnvironment(int argc,
				char *argv[],
				const std::string path,
				bool evaluate);

}  // namespace fast_dqn
#endif  // SRC_ENVIRONMENT_H_
