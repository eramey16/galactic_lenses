#!/bin/sh
for ((i=$1;i<$2;i++)); do
    sbatch runit$i
done