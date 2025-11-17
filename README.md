PGD MLOps Major Assignment

## Repository structure
- `main` : initial branch (README + .gitignore)
- `dev` : model development branch (train.py, test.py, CI workflow)
- `docker_cicd` : Flask app, Dockerfile, Kubernetes manifests

## Summary
This repository contains the work for the PGD MLOps Major Assignment:
- Training & testing scripts for a Decision Tree model on the Olivetti faces dataset.
- GitHub Actions CI to run training and testing on pushes to `dev`.
- A Flask app and Dockerfile (in `docker_cicd` branch) to serve the model.
- Kubernetes manifests to deploy the app with 3 replicas on Minikube.

## How to reproduce (high level)
1. Clone the repo.
2. Switch to `dev` branch for model work.
3. Build and push Docker image from `docker_cicd` branch, or load image into Minikube.
4. Apply Kubernetes manifests and expose the service.

## Author
Anant Utkarsh
