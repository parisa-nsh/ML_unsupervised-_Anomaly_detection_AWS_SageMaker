# SageMaker setup: prerequisites (Step 1)

You have an **AWS account** and an **S3 bucket**. This guide covers the rest so you can run the pipeline on SageMaker instead of locally.

---

## 1. S3 bucket (you have this)

- **Bucket name:** Use your existing bucket (e.g. `my-ml-bucket`). The launcher scripts will write/read under prefixes like `data/raw`, `data/features`, `models/autoencoder`, `data/scores`.
- **No special config needed.** Just note the bucket name; you’ll set `SAGEMAKER_BUCKET=your-bucket-name` when running the SageMaker steps.

---

## 2. IAM role for SageMaker

SageMaker runs Processing and Training jobs on your behalf. Those jobs need an **IAM role** that has:

- Permission for SageMaker to assume it (so jobs can start).
- Permission to access S3 (read/write your bucket).
- Permission to run as SageMaker (create/describe jobs, write logs, etc.).

### Option A – Create the role in the AWS Console

1. Open **IAM** → **Roles** → **Create role**.
2. **Trusted entity type:** AWS service.
3. **Use case:** Choose **SageMaker** (e.g. “SageMaker - Execution”).
4. Click **Next**. Attach these **managed policies** (or equivalent):
   - **AmazonSageMakerFullAccess** (easiest),  
     **or** a custom policy that allows at least:
     - `sagemaker:CreateProcessingJob`, `sagemaker:DescribeProcessingJob`, …
     - `sagemaker:CreateTrainingJob`, `sagemaker:DescribeTrainingJob`, …
     - `sagemaker:CreateModel`, and related Describe/List as needed.
   - **AmazonS3FullAccess** (for your bucket),  
     **or** a custom policy that allows `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on your bucket and its prefixes.
5. Click **Next** → name the role (e.g. `SageMakerExecutionRole-MLProject`) → **Create role**.
6. Open the role and copy the **Role ARN** (e.g. `arn:aws:iam::123456789012:role/SageMakerExecutionRole-MLProject`). You will set:
   ```bash
   set SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE_NAME
   ```
   (Use `export` on Linux/macOS.)

### Option B – Create the role with AWS CLI

Replace `YOUR_ACCOUNT_ID` and `YOUR_BUCKET_NAME` (and role name if you change it).

**Trust policy** (save as `trust-policy-sagemaker.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Create the role and attach AWS managed policies:

```bash
aws iam create-role --role-name SageMakerExecutionRole-MLProject --assume-role-policy-document file://trust-policy-sagemaker.json
aws iam attach-role-policy --role-name SageMakerExecutionRole-MLProject --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name SageMakerExecutionRole-MLProject --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

Then get the ARN:

```bash
aws iam get-role --role-name SageMakerExecutionRole-MLProject --query "Role.Arn" --output text
```

Set that as `SAGEMAKER_ROLE` (see above).

---

## 3. AWS CLI configured on your machine

The **launcher scripts** (`sagemaker/run_processing.py`, `run_training.py`, `run_batch_inference.py`) run on **your machine** and call the SageMaker (and S3) APIs. They need credentials.

- **Recommended:** Configure the AWS CLI so boto3/SDK use the default credential chain:
  ```bash
  aws configure
  ```
  Enter your **Access Key ID**, **Secret Access Key**, and default **region** (e.g. `us-east-1`).

- **Alternatively:** Set environment variables instead of `aws configure`:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_DEFAULT_REGION` (e.g. `us-east-1`)

The IAM user (or role) behind these credentials must be allowed to:
- Create/describe SageMaker Processing and Training jobs.
- Pass the SageMaker execution role to SageMaker (e.g. `iam:PassRole` for that role).
- Read/write S3 in your bucket (to upload code, data, and to let the job read/write).

---

## 4. Quick checks (no code)

- **S3:**  
  `aws s3 ls s3://YOUR_BUCKET_NAME/`  
  should list the bucket (or show “no objects”) without access denied.

- **Role ARN:**  
  You have the role ARN written down and will set `SAGEMAKER_ROLE` to it.

- **Credentials / region:**  
  `aws sts get-caller-identity`  
  and  
  `aws sagemaker list-training-jobs --max-results 1`  
  should succeed (no auth/region errors).

---

## 5. What you’ll set when running the pipeline on SageMaker

Before running the launcher scripts:

```bash
set SAGEMAKER_BUCKET=your-bucket-name
set SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_ROLE_NAME
```

(Use `export` on Linux/macOS.) Optionally set region if not using default:

```bash
set AWS_DEFAULT_REGION=us-east-1
```

No code changes are required for Step 1; the repo already uses these environment variables.

---

## 6. Service quotas (required in your region)

SageMaker **Processing** and **Training** jobs use ML instance types (e.g. `ml.m5.xlarge`). In some regions the default quota for these is **0**, so jobs fail with `ResourceLimitExceeded` until you request an increase.

### Quotas to request (for this project)

| Quota name | Used by | Request at least |
|------------|---------|-------------------|
| **ml.m5.xlarge for training job usage** | SageMaker Training Job | 2 |
| **ml.m5.xlarge for processing job usage** | Feature processing + batch inference | 2 |

### How to request an increase

1. In the AWS Console, open **Service Quotas** (search for “Service Quotas” in the top bar).
2. In the left menu, click **AWS services**.
3. In the service list, search for **SageMaker** and click **Amazon SageMaker** (do *not* search for the quota name on this page—that search filters *services*, not quotas).
4. Set the **region** (top right) to the one you use (e.g. **Canada (Central)**).
5. On the SageMaker quotas page, use **Search by quota name** and type **processing** or **ml.m5.xlarge** to find:
   - **ml.m5.xlarge for processing job usage** → Request quota increase → New limit value: **2**.
   - **ml.m5.xlarge for training job usage** → same steps if needed.
6. Submit each request. AWS typically responds within 1–2 business days.
7. Check status under **Service Quotas** → **Quota request history**.

Once both quotas are approved, the launcher scripts (`run_processing.py`, `run_training.py`, `run_batch_inference.py`) can run successfully in that region.
