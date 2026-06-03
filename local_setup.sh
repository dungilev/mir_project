#!/usr/bin/env bash
# Local environment setup for mir_project (auto-generated)
# Source this file to export recommended env vars for your laptop.

# MiR controller IP (adjust if needed)
export MIR_IP=${MIR_IP:-192.168.0.177}

# Inference device: 'cpu' or GPU index like '0'. Detected: GPU available
# Set to 'cpu' to force CPU.
export KHOANGCACH_DEVICE=${KHOANGCACH_DEVICE:-0}

# RealSense camera usage (0/1)
export KHOANGCACH_USE_REALSENSE=${KHOANGCACH_USE_REALSENSE:-0}

# ONNX inference usage (0/1)
export KHOANGCACH_USE_ONNX=${KHOANGCACH_USE_ONNX:-0}

# Inference frequency (process every N frames)
export KHOANGCACH_INFER_EVERY=${KHOANGCACH_INFER_EVERY:-2}

echo "Local env configured: KHOANGCACH_DEVICE=$KHOANGCACH_DEVICE, REALSENSE=$KHOANGCACH_USE_REALSENSE"
