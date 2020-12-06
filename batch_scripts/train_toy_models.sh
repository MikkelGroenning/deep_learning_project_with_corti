 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J train_toy_models
 #BSUB -n 1
 #BSUB -W 1:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Running script..."
 python3 src/models/example_models.py
