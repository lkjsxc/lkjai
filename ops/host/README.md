# Host Ops

## Purpose

This directory contains host setup helpers for machines that run Docker Compose
with NVIDIA GPU access.

## Contents

- [install-nvidia-container-toolkit.sh](install-nvidia-container-toolkit.sh):
  installs and configures NVIDIA Container Toolkit for Docker hosts.

## Rules

- Keep host mutation explicit and opt-in.
- Keep project build, test, training, and serving commands in Docker Compose.
