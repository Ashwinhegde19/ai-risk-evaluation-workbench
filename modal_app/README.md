# Modal OSS Endpoint

This folder contains the optional Modal deployment scaffold for hosted open-source assistant inference.

## Deploy

Install and authenticate Modal:

```bash
pip install modal
modal setup
```

Deploy the endpoint:

```bash
modal deploy modal_app/oss_endpoint.py
```

After deployment, copy the generated endpoint URL into `.env`:

```bash
OSS_BACKEND=modal
MODAL_OSS_ENDPOINT=https://your-modal-endpoint.example
MODAL_OSS_MODEL_NAME=Qwen/Qwen2.5-3B-Instruct
MODAL_TIMEOUT_SECONDS=120
```

## Runtime Model

The default Modal model is:

```txt
Qwen/Qwen2.5-3B-Instruct
```

You can override it during deployment with:

```bash
MODAL_MODEL_ID=Qwen/Qwen2.5-7B-Instruct modal deploy modal_app/oss_endpoint.py
```

The app-side client is `models.modal_endpoint.ModalEndpointClient`, and it expects the endpoint response to include a `response`, `text`, `content`, or OpenAI-style `choices` field.
