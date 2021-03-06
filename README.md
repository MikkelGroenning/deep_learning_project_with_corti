# Deep Learning Project with Corti
Deep learning project in course 02456 Deep Learning at DTU in collaboration with Corti.

1. Main implementation and scripts are found in the `src` directory, which can be installed as a python module using `pip install -e .`, ideally from a virtual environment using python 3.6.
2. Model defitions are found in the `src/models/` directory
3. This project uses a MADE implemementation found in https://github.com/ritchie46/vi-torch
4. Trained models are stored in the `models/` directory
5. Various notebooks used throughout the project can be found in the `notebooks/` directory.
6. The models are trained on the DTU HPC Cluster using the batch scripts in `batch_scripts/`.

## Short installation guide

1. Download/clone repository
2. From the repository root, create a virtual environment through the `conda env create -n dl-corti` command 
3. Activate conda environment using `conda activate dl-corti`
4. Download and unzip `models.zip` and `data.zip` files (link found in report appendix) and replace the corresponding folders in the repository root.
5. Run eg. `python src/reconstructions.py`