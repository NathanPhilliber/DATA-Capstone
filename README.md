# NASA: DATA Capstone
DATA-451

**Steven Bradley, Nathan Philliber, Mateo Ibarguen**

Docker Instructions
--
1. ssh into AWS server
2. cd to top-level source code directory
3. Build the Docker image via `./docker_rebuild.sh`. This will create a Docker image named: `nasa/peak_detection:latest`.
4. Now that the image exists, you should be able to execute Python code by running the following: `./run_docker.sh <python_script> <args>`

## Quickstart
In order to execute these scripts, you will need to have at least `python3.6` installed and all the [dependencies](requirements.txt).

We added an example dataset and a pre-trained model in order to demonstrate how our project works. When you are asked to select a dataset, make sure to select: `example_set`. When prompted to select a model, make sure to select: `GoogleModel_BC-1234-50c-all.0513.2202` 
```bash
source virtualenv/bin
```

- In order to train the model with the additional example dataset, type the following:
```
python3 -m models.run_train continue
```
- In order to evaluate the model with the example dataset, type the following:
```
python3 -m models.run_train evaluate
```

![roc_curve](data/results/GoogleModel_BC-1234-50c-all.0513.2202/eval/roc_curve-0517.1722.png)

## S3 Integration
Due to the size of the datasets that are generated, we implemented a way for the datasets to be stored in S3 rather than on disk. If this is something that you would like to use, please contact us with your AWS account number and we can grant you access to S3.

*Note: The dataset that you are working with will be downloaded form S3, so it will still end up on disk. This was implemented to save space when dealing with multiple datasets.*

By default, data that is generated will be saved to disk. To use S3 instead, uncomment this line:
```
spectra_generator = S3SpectraGenerator(s3_dir, matlab_script=matlab_script, nc=num_channels, n_max=n_max, n_max_s=n_max_s, scale=scale, omega_shift=omega_shift, dg=dg, dgs=dgs)
``` 
and comment out 
```
spectra_generator = LocalSpectraGenerator(matlab_script=matlab_script, nc=num_channels, n_max=n_max, n_max_s=n_max_s,
                                             scale=scale, omega_shift=omega_shift, dg=dg, dgs=dgs, save_dir=directory)
```
 in *run_gen.py*
