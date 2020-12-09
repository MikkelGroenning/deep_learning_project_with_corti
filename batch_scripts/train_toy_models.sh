 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J train_toy
 #BSUB -n 1
 #BSUB -W 2:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=8GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Training models"
 python3 src/models/toy_models.py
 echo "Plotting results"
 python3 src/visualization/toy.py
