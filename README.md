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

![eval][data/results/GoogleModel_BC-1234-50c-all.0513.2202/eval/roc_curve-0517.1722.png]
