 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J WordRAE
 #BSUB -n 1
 #BSUB -W 5:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=16GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Running script..."
 python3 src/models/word_models.py WordRAE
