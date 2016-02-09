#include "fast_dqn.h"
#include "environment.h"
#include <minecraft_dqn_interface.cpp>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <cmath>
#include <iostream>
#include <deque>
#include <algorithm>
#include <boost/filesystem.hpp>

DEFINE_bool(verbose, false, "verbose output");
DEFINE_bool(gpu, true, "Use GPU to brew Caffe");
DEFINE_int32(device, -1, "Which GPU to use");
DEFINE_bool(gui, false, "Open a GUI window");
DEFINE_string(rom, "breakout.bin", "Atari 2600 ROM to play");
DEFINE_string(solver, "models/fast_dqn_solver.prototxt", "Solver parameter"
  "file (*.prototxt)");
DEFINE_int32(memory, 500000, "Capacity of replay memory");
DEFINE_int32(explore, 1000000, "Number of iterations needed for epsilon"
  "to reach 0.1");
DEFINE_double(gamma, 0.95, "Discount factor of future rewards (0,1]");
DEFINE_int32(memory_threshold, 100, "Enough amount of transitions to start "
  "learning");
DEFINE_int32(skip_frame, 3, "Number of frames skipped");
DEFINE_bool(show_frame, false, "Show the current frame in CUI");
DEFINE_string(model, "", "Model file to load");
DEFINE_bool(evaluate, false, "Evaluation mode: only playing a game, no "
  "updates");
DEFINE_string(snapshot, "", "The solver state to load (*.solverstate).");
DEFINE_bool(resume, false, "Automatically resume training from the latest snapshot.");
DEFINE_double(evaluate_with_epsilon, 0.05, "Epsilon value to be used in "
  "evaluation mode");
DEFINE_double(repeat_games, 1, "Number of games played in evaluation mode");
DEFINE_int32(steps_per_epoch, 5000, "Number of training steps per epoch");

double CalculateEpsilon(const int iter, const int memory) {
  // if resume, try to initialize the memory with good states
  if (FLAGS_resume && memory < FLAGS_memory_threshold) {
    return 0.05;
  }

  if (iter < FLAGS_explore) {
    return 1.0 - 0.9 * (static_cast<double>(iter) / FLAGS_explore);
  } else {
    return 0.1;
  }
}

/**
 * Play one episode and return the total score
 */
double PlayOneEpisode(
    fast_dqn::EnvironmentSp environmentSp,
    fast_dqn::Fast_DQN* dqn,
    const double epsilon,
    const bool update) {
  assert(!environmentSp->EpisodeOver());
  std::deque<fast_dqn::FrameDataSp> past_frames;
  auto total_score = 0.0;
  for (auto frame = 0; !environmentSp->EpisodeOver(); ++frame) {
    if (FLAGS_verbose)
      LOG(INFO) << "frame: " << frame;
    const auto current_frame = environmentSp->PreprocessScreen();
//     if (FLAGS_show_frame) {
//       std::cout << fast_dqn::DrawFrame(*current_frame);
//     }
    past_frames.push_back(current_frame);
    if (past_frames.size() < fast_dqn::kInputFrameCount) {
      // If there are not past frames enough for DQN input, just select NOOP
      environmentSp->ActNoop();
    } else {
      if (past_frames.size() > fast_dqn::kInputFrameCount) {
        past_frames.pop_front();
      }
      fast_dqn::State input_frames;
      std::copy(past_frames.begin(), past_frames.end(), input_frames.begin());
      const auto action = dqn->SelectAction(input_frames, epsilon);

      auto immediate_score = environmentSp->Act(action);
      total_score += immediate_score;

      // Rewards for DQN are normalized as follows:
      // 1 for any positive score, -1 for any negative score, otherwise 0
      const auto reward =
          immediate_score == 0 ?
              0 :
              immediate_score /= std::abs(immediate_score);

      if (update) {
        // Add the current transition to replay memory
        const auto transition = environmentSp->EpisodeOver() ?
            fast_dqn::Transition(input_frames, action, reward, nullptr) :
            fast_dqn::Transition(
                input_frames,
                action,
                reward,
                environmentSp->PreprocessScreen());
        dqn->AddTransition(transition);
        // If the size of replay memory is enough, update DQN
        if (dqn->memory_size() >= FLAGS_memory_threshold) {
          dqn->Update();
        }
      }
    }
  }
  environmentSp->Reset();
  return total_score;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  google::LogToStderr();

  if (FLAGS_gpu) {
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    if (FLAGS_device >= 0) {
      caffe::Caffe::SetDevice(FLAGS_device);
    }
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }


  fast_dqn::EnvironmentSp environmentSp =
  fast_dqn::CreateEnvironment(argc, argv, "dependencies/minecraft_dqn_interface/", FLAGS_evaluate);

  // Get the vector of legal actions
  const fast_dqn::Environment::ActionVec legal_actions = 
    environmentSp->GetMinimalActionSet();

  fast_dqn::Fast_DQN dqn(environmentSp, legal_actions, FLAGS_solver, 
                         FLAGS_memory, FLAGS_gamma, FLAGS_verbose);

  dqn.Initialize();

  boost::filesystem::path save_path("model/");
  // Look for a recent snapshot to resume
  if (FLAGS_resume && FLAGS_snapshot.empty()) {
    FLAGS_snapshot = fast_dqn::FindLatestSnapshot(save_path.native());
    std::cout << "snapshot: " << FLAGS_snapshot << std::endl;
  }

  if (!FLAGS_snapshot.empty()) {
    boost::filesystem::path p(FLAGS_snapshot);
    p = p.parent_path() / p.stem();
    //std::string mem_fname = p.native() + ".replaymemory";
    //CHECK(boost::filesystem::is_regular_file(mem_fname))
    //  << "Unable to find .replaymemory for snapshot: " << FLAGS_snapshot;

    std::string model_fname = p.native() + ".caffemodel";
    CHECK(boost::filesystem::is_regular_file(model_fname))
      << "Unable to find .caffemodel for snapshot: " << FLAGS_snapshot;

    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    dqn.RestoreSolver(FLAGS_snapshot);
    //dqn.LoadReplayMemory(mem_fname);
    dqn.LoadTrainedModel(model_fname);
  }

  if (!FLAGS_model.empty()) {
    // Just evaluate the given trained model
    LOG(INFO) << "Loading " << FLAGS_model;
    dqn.LoadTrainedModel(FLAGS_model);
  }

  if (FLAGS_evaluate) {
    CHECK(!FLAGS_model.empty()) << "Must provide model file to run with evaluate";
    auto total_score = 0.0;
    for (auto i = 0; i < FLAGS_repeat_games; ++i) {
      LOG(INFO) << "game: ";
      const auto score =
          PlayOneEpisode(environmentSp, &dqn, FLAGS_evaluate_with_epsilon, false);
      LOG(INFO) << "score: " << score;
      total_score += score;
    }
    LOG(INFO) << "total_score: " << total_score;
    return 0;
  }

  double total_score = 0.0;
  double epoch_total_score = 0.0;
  int epoch_episode_count = 0.0;
  double total_time = 0.0;
  int next_epoch_boundry = FLAGS_steps_per_epoch;
  double running_average = 0.0;
  double plot_average_discount = 0.05;

  std::ofstream training_data(".//training_log.csv");
  training_data << FLAGS_rom << "," << FLAGS_steps_per_epoch
    << ",,," << std::endl;
  training_data << "Epoch,Epoch avg score,Hours training,Number of episodes"
    ",episodes in epoch" << std::endl;


  for (auto episode = 0;; episode++) {
    caffe::Timer run_timer;
    run_timer.Start();

    epoch_episode_count++;
    const auto epsilon = CalculateEpsilon(dqn.current_iteration(), dqn.memory_size());
    auto train_score = PlayOneEpisode(environmentSp, &dqn, epsilon, true);

    epoch_total_score += train_score;
    if (dqn.current_iteration() > 0)  // started training?
      total_time += run_timer.MilliSeconds();
    LOG(INFO) << "training score(" << episode << "): "
      << train_score << std::endl;

    if (episode == 0)
      running_average = train_score;
    else
      running_average = train_score*plot_average_discount
        + running_average*(1.0-plot_average_discount);

    if (dqn.current_iteration() >= next_epoch_boundry) {   
      double hours =  total_time / 1000. / 3600.;
      int epoc_number = static_cast<int>(
        (next_epoch_boundry)/FLAGS_steps_per_epoch);
      LOG(INFO) << "epoch(" << epoc_number
        << ":" << dqn.current_iteration() << "): "
        << "average score " << running_average << " in "
        << hours << " hour(s)";

      if (dqn.current_iteration()) {
        auto hours_for_million = hours/(
          dqn.current_iteration()/1000000.0);
        LOG(INFO) << "Estimated Time for 1 million iterations: "
          << hours_for_million
          << " hours";
      }

      training_data << epoc_number << ", " << running_average << ", " << hours
        << ", " << episode << ", " << epoch_episode_count << std::endl;

      epoch_total_score = 0.0;
      epoch_episode_count = 0;

      while (next_epoch_boundry < dqn.current_iteration())
        next_epoch_boundry += FLAGS_steps_per_epoch;
    }
  }

  training_data.close();
}

