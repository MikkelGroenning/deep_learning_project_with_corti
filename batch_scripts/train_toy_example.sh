 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J toy_example
 #BSUB -n 1
 #BSUB -W 10:00
 #BSUB -R "rusage[mem=32GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Running script..."
 python3 src/models/train_toy_example.py
