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

  static constexpr auto kCroppedVolumeSize = 10;
  static constexpr auto kCroppedVolumeDataSize = 
    kCroppedVolumeSize * kCroppedVolumeSize * kCroppedVolumeSize;
  
  static constexpr auto kTransformDataSize = 5;
  
  static constexpr auto kInputCount = 4;
  static constexpr auto kInputVolumeDataSize = 
    kCroppedVolumeDataSize * kInputCount;
  static constexpr auto kInputTransformDataSize = 
    kTransformDataSize * kInputCount;

  using VolumeData = std::array<uint8_t, kCroppedVolumeDataSize>;
  using VolumeDataSp = std::shared_ptr<VolumeData>;
  
  using TransformData = std::array<float, kTransformDataSize>;
  using TransformDataSp = std::shared_ptr<TransformData>;

  using VolumeState = std::array<VolumeDataSp, kInputCount>;
  using TransformState = std::array<TransformDataSp, kInputCount>;

  virtual VolumeDataSp PreprocessScreen() = 0;
  
  virtual TransformDataSp GetTransform() = 0;

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
