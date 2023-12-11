#!/bin/bash

NUM=8

mpirun -np $NUM ~/Documents/Incompact3d/xcompact3d &>> log.x3d
# ~/Documents/Incompact3d/xcompact3d &> log.x3d
