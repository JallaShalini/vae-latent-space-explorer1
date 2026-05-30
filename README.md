# VAE Latent Space Explorer

This repository contains a Variational Autoencoder (VAE) trained on MNIST and a Streamlit application for exploring the learned latent space.

Quick start (local):

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
venv\Scripts\python -m pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```powershell
docker compose up --build -d
# then open http://localhost:8501
```

Deployment options:
- Streamlit Community Cloud: push the repository to GitHub and use Streamlit Cloud to deploy the app directly. The app entrypoint is `app/app.py` and Streamlit Cloud will auto-detect it.
- Docker / GHCR: This repo includes a GitHub Actions workflow (`.github/workflows/publish-image.yml`) that will build and publish a Docker image to GitHub Container Registry (GHCR) when you push to `main`/`master`.

Notes about the CI workflow:
- The workflow logs in to `ghcr.io` using `${{ secrets.GITHUB_TOKEN }}` and tags images as `ghcr.io/<owner>/<repo>:latest` and with the commit SHA. Open the Packages / Container registry in your GitHub repository to find the published image.

If you want me to push an image to DockerHub or set up automatic Streamlit Cloud deploys, I can add the workflow, but you'll need to provide DockerHub credentials or the Streamlit deploy token.# VAE Latent Space Explorer

Project scaffold for a Variational Autoencoder built with PyTorch and explored through a Streamlit app.

## Phase 0 Setup

Create a local virtual environment:

```bash
py -3 -m venv venv
```

Install dependencies:

```bash
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
```

Run the app with Docker:

```bash
docker compose up --build -d
```

The application is configured to listen on port `8501` by default. Adjust values in `.env` if needed after copying from `.env.example`.

## Next Phases

The remaining phases will add the VAE model, training pipeline, evaluation scripts, and latent-space explorer pages.
