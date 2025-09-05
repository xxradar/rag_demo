# Security Summary & Recommendations

## Key Findings
- Secrets Handling: `.env` is gitignored; keep real keys in env/secret managers and never log them.
- Cloud Credentials: Uses long‑lived AWS keys via env vars; prefer IAM roles or short‑lived STS tokens; scope to the target S3 bucket.
- Data Egress & Privacy: Document chunks are sent to OpenAI for embeddings/chat; confirm data‑processing approvals or use private/enterprise endpoints.
- Local Persistence: `chroma_db/` stores raw text and embeddings unencrypted; keep on encrypted volumes and isolate per dataset/environment.
- Multitenancy Boundary: Single `documents` collection risks data mixing; include partitioning metadata or per‑env collection names.
- Logging Exposure: Logs show S3 keys and text previews; treat logs as sensitive and gate verbose output.
- Untrusted Inputs: PDF parsing (PyPDF2) handles untrusted files; add file size/type limits and malware scanning if needed.
- Temporary Files: S3 downloads are cleaned up; ensure temp dirs are on encrypted disks and not world‑readable.
- Exception Handling: Broad `except` blocks can hide issues; log safely with enough detail for triage.
- Dependencies & SBOM: Many pinned packages; audit regularly and trim unused deps.

## Actionable Recommendations
- Secrets: Use AWS Secrets Manager/SSM and IAM roles; forbid committing `.env`.
- Isolation: Partition Chroma by bucket/env (e.g., `documents-<env>-<bucket>`); add `dataset_id` metadata.
- Logging: Default to quiet mode; remove text previews unless `--verbose` is set.
- Data Controls: Enforce max file size, allowed extensions, and optional content scanning.
- At Rest: Store `chroma_db/` on encrypted volumes; restrict filesystem permissions.

## Suggested Audits (local/CI)
```bash
pip install pip-audit && pip-audit
pip install detect-secrets && detect-secrets scan
trivy fs --scanners vuln,secret .
gitleaks detect --no-git
```

## Least-Privilege Policy (S3)
- Limit to specific bucket/prefix.
- Allow only `s3:ListBucket` and `s3:GetObject`.
- Deny write/delete unless explicitly required by a workflow.

