#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/facenet/without_maxout_solver.prototxt --gpu=1

