#!/bin/bash

NUM=8

cd "$(dirname "$0")"

mpirun -np $NUM xcompact3d &>> log.x3d
# ~/Documents/Incompact3d/xcompact3d &> log.x3d
