 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J train_vrae_words
 #BSUB -n 1
 #BSUB -W 5:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=16GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Running script..."
 python3 src/models/vrae_words.py
