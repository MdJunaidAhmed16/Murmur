# Deploying the public demo

**Fastest path (no local install): Google Cloud Shell.** After steps 1, 3–4 and 6 below, open
Cloud Shell (the `>_` icon in the console) and run:

```bash
git clone https://github.com/MdJunaidAhmed16/Murmur.git && cd Murmur
PROJECT=$(gcloud config get-value project) GPU=0 bash scripts/deploy_cloudrun.sh   # CPU, no quota
# later, once the L4 quota is granted:
PROJECT=$(gcloud config get-value project) GPU=1 bash scripts/deploy_cloudrun.sh
```

`GPU=0` runs Whisper `base.en` and Kokoro on 4 vCPUs: no GPU quota needed, slower to first
audio. `GPU=1` uses one L4.

The demo runs on **Cloud Run with one NVIDIA L4 GPU**. Whisper and Kokoro run on the GPU;
the LLM is **Claude through OpenRouter** (`MURMUR_BACKEND=hosted`). It scales to zero when
nobody is using it, so an idle demo costs nothing. The trade-off is a cold start: the first
visitor after a quiet spell waits while the container starts and loads the models.

Each browser gets its **own** memory store (`MURMUR_DEMO_MODE=true`), so visitors never hear
each other's facts. The stores live on the container's disk and are wiped when it scales to
zero. That is deliberate for a public demo, and the page says so.

Everything below is one-time setup except step 8. Commands run in a WSL terminal.

## 1. Project and billing

1. Go to <https://console.cloud.google.com/projectcreate> and create a project, e.g.
   `murmur-demo`. Note the **project ID** (it may get a numeric suffix).
2. Link a billing account: <https://console.cloud.google.com/billing>. GPUs are not available
   on the free trial's credits until billing is upgraded to a paid account.
3. **Set a budget alert before anything else:** Billing → Budgets & alerts → Create budget,
   e.g. $20/month with email alerts at 50%, 90% and 100%. A GPU left running by mistake is
   the main way this project can cost real money.

## 2. Install and log in to gcloud (WSL)

```bash
curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
exec -l $SHELL
gcloud init                     # log in in the browser, pick the project
gcloud auth application-default login
```

## 3. Enable the APIs

```bash
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com secretmanager.googleapis.com
```

## 4. A registry for the image

```bash
gcloud artifacts repositories create murmur \
  --repository-format=docker --location=us-central1
```

Use the same region everywhere. `us-central1` offers L4 GPUs on Cloud Run; if you pick another
region, check it does: <https://cloud.google.com/run/docs/configuring/services/gpu>.

## 5. GPU quota

Console → IAM & Admin → **Quotas & System Limits**, filter for *Cloud Run Admin API* and
**"Total Nvidia L4 GPU allocation without zonal redundancy, per project per region"** in your
region. If the value is 0, select it → **Edit quota** → request 1, with a one-line reason
("demo of a voice agent, one instance"). Approval can take from minutes to a couple of days;
nothing else here waits on it until step 8.

## 6. OpenRouter key, as a secret

1. Create a key at <https://openrouter.ai/settings/keys>. **Set a credit limit on the key**
   (e.g. $5) so a public demo cannot run up a bill.
2. Store it in Secret Manager (it never goes into the image or the repo):

   ```bash
   read -rs KEY && printf %s "$KEY" | gcloud secrets create openrouter-api-key --data-file=-
   ```

3. Let Cloud Run read it:

   ```bash
   PROJECT_NUMBER=$(gcloud projects describe "$(gcloud config get-value project)" \
     --format='value(projectNumber)')
   gcloud secrets add-iam-policy-binding openrouter-api-key \
     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --role=roles/secretmanager.secretAccessor
   ```

## 7. Choose the model (optional)

The default is `anthropic/claude-opus-5.5`. For the lowest time to first token in a voice
demo, `anthropic/claude-haiku-5.5` is the one to compare it against (R4 measures both). Pass
`MODEL=...` to the deploy script to change it.

## 8. Deploy

```bash
cd ~/murmur
PROJECT=<your-project-id> bash scripts/deploy_cloudrun.sh
```

The first build takes a while: Cloud Build installs the CUDA wheels and downloads the models
on Google's network. Your machine only uploads the source. Later builds reuse the dependency
layer. The script prints the URL at the end.

## Checking it

- Open the URL in Chrome or Edge and wait for the first load (cold start).
- Logs: `gcloud run services logs read murmur --region us-central1 --limit 50`. Look for
  `"event": "ready"` with `"tts_providers": ["CUDAExecutionProvider", ...]` and
  `"backend": "hosted"`.
- A second person connecting while someone is talking gets "someone else is talking to it".
  One GPU, one conversation (CLAUDE.md scope).

## Turning it off

Scale-to-zero means it costs nothing while idle. To take it offline completely:

```bash
gcloud run services delete murmur --region us-central1
```

## What it costs

Pay-per-second while an instance is running (GPU, 4 vCPU, 16 GiB), plus OpenRouter tokens per
conversation, plus a little Artifact Registry storage for the image. Check current prices at
<https://cloud.google.com/run/pricing> before leaving it public; the budget alert from step 1
is the safety net.
