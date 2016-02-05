#! /bin/sh

# given arg that is name of test
mkdir -p tests/$1/models

cp training_log.csv train_progress.png models/fast_dqn*.prototxt tests/$1
cp model/* tests/$1/models
cp -R screenshots/ tests/$1/
