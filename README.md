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

We added an example dataset and a pre-trained model in order to demonstrate how our project works. 
```bash
source virtualenv/bin
```
