# ICE-RAFT
<img src="FlowChart_Visualization.png">

Deep learning-based optical flow tool for full-field analysis of deformation in radar imagery. This work builds upon the original work of RAFT (Teed and Deng, [2020](https://doi.org/10.1007/978-3-030-58536-5_24)) and the idea of differentiable warm-starting presented with E-RAFT (Gehrig, et al. [2021](https://doi.org/10.1109/3DV53792.2021.00030)) to consider a temporal multiresolution tree based on different temporal resolutions. The increased complexity enables higher accuracy in estimates and stability between different time steps needed in scientific analysis.


When using this code in research, please cite the following publication:
```bibtex
@InProceedings{UusinokaDeep2024,
  author = {Matias Uusinoka, Jari Haapala, and Arttu Polojärvi},
  title = {Deep learning-based optical flow in fine-scale deformation mapping of sea ice dynamics},
  year = {2024}
}
```

## Requirements
The algorithm requires mainly common libraries such as Numpy, Scipy, PyTorch, Torchvision and Numba. A conda environment with the appropriate release versions can easily be installed based on the provided .yml file:
```Shell
conda env create -f environment.yml
```

## Demo
ICE-RAFT can easily be used with the default models provided by [Pytorch](https://pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_large.html#torchvision.models.optical_flow.raft_large). Using a specific model can be indicated with the algorithm arguments.

The algorithm can be demoed by first running Displacement_Production.py with appropriate arguments, e.g.:
```Shell
python3 Displacement_Production.py --update_iterations 30 --image_resolution 1000 --result_file_name demoing_iceraft
```
Although the model can be run on CPUs, the user is advised to use GPUs. This is to avoid program crashing due to on-chip memory limitations.
After producing the displacement fields, Trajectories_Strains_Deformation.py can be run (on CPUs) similarly, e.g.:
```Shell
python3 Trajectories_Strains_Deformation.py --source_file demo_data/displacements/demoing_iceraft.npy --saved_quantities all --pixel_to_metric 10 --resolution_ratio 1
```
With larger datasets the processing is advised to be divided for parallelization with the multiprocessing library as indicated in the code.

Alternatively, shell scripts similar to Initialize_Deformations.sh and Initialize_Displacements.sh can be run. The example scripts have been tested on Slurm workload manager and with NVIDIA V100 GPUs.

## Verification
The algorithm can be tested against ground truth trajectories generated with Producing_verification_groundtruths.py. In Uusinoka, et al. (2024) the performance of the algorithm was tested based on end-point-error estimates between the ground truth and optical flow estimate trajectories.

The required files for similar performance testing can be found in the 'verification_data' folder, which can be indicated with the '--image_path' argument for Displacement_Production.py. Performance testing should be done with GPUs for sufficient accuracy.


## Arguments

### Displacement_Production.py
```--custom_model``` : Enables the use of pre-trained custom models.

```--image_path``` : Indicate the path to the images to be analyzed.

```--result_file_name``` : Giving a name to the created displacement file.

```--image_resolution``` : At what resolution will the images be analyzed?

```--displacement_resolution``` : Define temporal resolution for producing the final displacements.

```--temporal_scales``` : Define resolutions based on which the temporal multuresolution tree will be formulated.

```--update_iterations``` : The number of iterations done by each layer's update operator in the resolution tree.


### Trajectories_Strains_Deformation.py
```--source_file``` : Give the path to and the name of (with the file extension) the displacement file.

```--save_path``` : Give the path for saving the results.

```--output_name``` : Give the output filenames a prefix.

```--saved_quantities``` : Define which quantities will be saved. Options: "trajectories", "strain_tensors", "deformations", and "all".

```--spatial_scale``` : Spatial scale defines the size of the deformations objects. 2 is pixel scale, 3 the scale of two pixels etc.

```--pixel_to_metric``` : The spatial scale of one pixel i.e. how many meters does one pixel represent.

```--resolution_ratio``` : Define the resolution ratio between the original images and the displacement fields.

```--use_finite_strain``` : The deformations will either use infinitesimal or finite strain. Argument "TRUE" corresponds to using finite strains.

```--remove_displacements``` : Removing files in path data/displacements/ will save memory.


### verification_data.Producing_verification_groundtruths.py
```--motion_field``` : Choose which motion field to produce. The option are 'rigidbody_rotated', 'uniform_shear', 'uniform_shear_rotated', 'localized_shear', 'localized_shear_rotated'.

```--save_path``` : Give the folder for the save file.
  
```--scale``` : Choose the scale of the displacements from either 'small' or 'large.

```--num_tsteps``` : Choose the number of timesteps considered in the verification.
