# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Simplito sp. z o.o.

"""DeepFellow Suite provisioning, run by the chart's provisioning Job.

    python -m deepfellow_provision reconcile | verify | status

It is idempotent, fails closed and never logs a secret:

* The Kubernetes state Secret is the source of truth. Every run reads it first; when it already
  holds organization-id, project-id and project-api-key, nothing is created again and later steps
  only reconcile desired state.
* The Server returns the project API key exactly once, so it is written to the state Secret before
  anything else happens. If that write fails after the key was minted, the run stops for manual
  recovery instead of letting a later run mint a second key.
* Only ids and step outcomes are logged, never a credential.

It runs inside the Server image (python, and the Server package for create_admin) and uses only the
standard library, so it does not depend on the image's other packages. The Kubernetes API is reached
with the pod's ServiceAccount token, whose Role can touch nothing but the state Secret.
"""
