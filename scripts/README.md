# Infrastructure Scripts

## Setting up

Google Colab can be easily integrated with an existing Google Cloud Platform project. One can use their own VM instance for computing, and GCS buckets for storing large amounts of train and test data.

### Storage
More info about the support of external data in Colab is available [here](https://colab.research.google.com/notebooks/io.ipynb). 

TLDR; one can use Google Drive, Google Sheets, and Google Cloud Storage (GCS buckets), volume mounts are not supported.

### VM
Chosen VM instance type: `n1-standard-4`
Zone: `us-central1-a`

EU-based instances are a bit pricier, but latency should not be a concern for our purposes, so the US zone works just fine.

### How?
**Prerequisites:**
1. `gcloud` CLI tool
1. `gsutil` CLI tool

We will use the script `init.sh`.
1. Login into GCP
    ```bash
    gcloud auth login
    ```
1. Adjust the environment variables, most importantly the GCP project ID.
1. Run the script
    ```bash
    ./init.sh
    ```

## Managing the VM Instance
The `vm.sh` script can be used for starting and stopping the VM instance.

### Start
```bash
./vm.sh
# or
./vm.sh --start
```

### Stop
```bash
./vm.sh --stop
```

## Uploading New Data Into the GCS Bucket
Run the following command in order to upload new contents of `data-generator/data` into the bucket used by Colab:
```bash
gsutil -m rsync -r -x ".*" -C "../data-generator/data" gs://"${BUCKET_NAME}"
```
Where `BUCKET_NAME=<PROJECT_ID>-asl-bucket`.
The command does not re-upload already existing files.

The command is also run when the VM instance is started.

## Tear-down
When the VM instance and GCS bucket are no longer needed, they can be removed by running the following script:
```bash
./destroy.sh
```
