#!/bin/bash

args="$*"
export PYTHONPATH=$PYTHONPATH:/app/src
/bin/bash -c "python /app/src/autograd/run.py $args"