 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J CharacterVRAE
 #BSUB -n 1
 #BSUB -W 8:00
 #BSUB -B
 #BSUB -N
 #BSUB -R "rusage[mem=16GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 
 echo "Running script..."
 python3 src/models/character_models.py CharacterVRAE
