#!/bin/bash

NUM=8

cd "$(dirname "$0")"
pwd

mpirun -np $NUM ~/Documents/Incompact3d/xcompact3d &> log.x3d
# ~/Documents/Incompact3d/xcompact3d &> log.x3d
