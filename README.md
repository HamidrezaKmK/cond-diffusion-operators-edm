# EDM in function spaces (edm-fs)

This is the public code release of Christopher Beckham's internship at NVIDIA in using neural operators for time series modelling of climate data. This repository is based on [EDM](https://github.com/NVlabs/edm/).

## Setup

This repository is based on the EDM codebase and as such will have a similar set of requirements. First create a conda environment. We will use the `environment.yml` file that is in the root directory of this repository. This is the same as the corresponding environment file in the original EDM repository from Karras et al located [here](https://github.com/NVlabs/edm).

To create an environment called `edm_fs`, run:

```
conda env create -f environment.yml
```

**Note**: This will also contain the `neuraloperators` library implemented [here](https://github.com/neuraloperator/neuraloperator).

### Environment variables

Also cd into `exps` and copy `cp env.sh.bak env.sh` and define the following env variables:

- `$SAVE_DIR`: some location where experiments are to be saved.
- `$DATA_DIR`: this should be kept as is, since it points to the raw CWA/ERA dataset zarr file.

### Navier Stokes Dataset

We use a 2D Navier-Stokes dataset which consists of trajectories in time. Concretely, the neural operator diffusion model is trained to model the following conditional distribution `p(u_t | y_{t-k}, ..., y_{t+k})` where:

- `u` is the function to model (whose samples are at a fixed discretisation which is defined by `resolution` in `training.datasets.NSDataset`, e.g. `128` for 128px on both spatial dimensions);
- `y` are samples from the same function at a much coarser discretisation, and this is determined by `lowres_scale_factor` in `training.datasets.NSDataset`, i.e. if `lowres_scale_factor=0.0625` then this is a 16x reduction in spatial resolution).
- `k` denotes the size of the context window.

Whatever `$DATA_DIR` is defined to, cd into that directory and download the data:

```
wget https://zenodo.org/records/7495555/files/2D_NS_Re40.npy
```

Visit `notebooks/irregular_ns` for an example of how to load and visualize this dataset and make it irregular as well.

## Running experiments

Example:

```bash
python train.py --savedir=outputs/exps/example --cfg=./conf/example_ns.json
```

### Generation

To generate a trajectory using a pretrained model, we can use the `generate_samples.sh` script in `exps`. This is a convenience wrapper on top of `generate.py` and it is run with the following arguments:

```
bash generate_samples.sh \
  <experiment name> \
  <n diffusion steps> \
  <n traj> \
  <output file> [<extra args>]
```

Respectively, these arguments correspond to:

- The name of the experiment, _relative_ to `$SAVE_DIR`;
- number of diffusion steps to perform (higher is better quality but takes longer);
- length of the trajectory to generate;
- and output file.

Please see `generate.py` for the full list of supported arguments, and see the next section for an example of how to generate with this script.

## Pretrained models

An example pretrained model can be downloaded [here](https://drive.google.com/file/d/1lpH6WVPqjZU1qNCH_2aWejU834mo6Urj/view?usp=drive_link). Download it to `$SAVE_DIR` and untar it via:

```
cd $SAVE_DIR && tar -xvzf test_ns_ws3_ngf64_v2.tar.gz
```

To generate a sample file from this model for 200 diffusion timesteps, cd into `exps` and run:

```
bash generate_samples.sh \
  test_ns_ws3_ngf64_v2/4148123 \
  200 \
  64 \
  samples200.pt
```

This will spit out a file called `samples200.pt` in the same directory. To generate a 30-fps video from these samples, run the following:

```
bash generate_video_from_samples.sh \
  samples200.pt \
  30 \
  samples200.pt.mp4
```

Example video (in gif format):

![generated viz](media/generated.gif)

with the first column denoting ground truth `u_t` (it is the same across each row), middle column denoting the generated function from diffusion `\tilde{u_t}`, and the third column denoting the low-res function `y_t` (again, same for each row).

### Super-resolution

Since this model was trained with samples from `u` being 64px, we can perform 2x super-resolution by passing in `--resolution=128` like so:

```
bash generate_from_samples.sh \
  test_ns_ws3_ngf64_v2/4148123 \
  200 \
  samples200_128.pt \
  --resolution=128 --batch_size=16
```

Example video (in gif format):

![generated viz](media/generated128.gif)

## Bugs / limitations

This section details some things that should be improved or considered by whomever forks this repository.

### Generation

If you make posthoc changes to the model code (e.g. `training.networks.py`) and then want to generate samples you should also add `--reload_network`, e.g

```
bash generate.sh ... --reload_network
```

This will tell the generation script to instead instantiate the model with its network definition as defined in `networks` and then load the weights from the pickle. By default, EDM's training script pickles not just the model weights but also the network code, and this can be frustrating if one wants to make post-hoc changes to the code which are backward compatible with existing pretrained models.

### Training

Neural operators require significantly more parameters than their finite-d counterparts and this issue is also exacerbated when one is training high-res diffusion models. I suggest future works look at latent consistency models, e.g. performing function space diffusion in the latent space of a pretrained autoencoder. Otherwise, the code should be modified to support `float16` training to alleviate the memory burden.

## Credits

Thanks to my co-authors Kamyar Azzizadenesheli, Nikola Kovachki, Jean Kossaifi, Boris Bonev, and Anima Anandkumar. Special thanks to Tero Karras, Morteza Mardani, Noah Brenowitz, and Miika Aittala.
