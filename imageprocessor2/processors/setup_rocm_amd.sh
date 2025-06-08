#!/bin/bash

set -e

echo "🧼 Cleaning up any old ROCm repos..."
sudo rm -f /etc/apt/sources.list.d/rocm.list

echo "🔑 Adding ROCm 5.7 repo key and source..."
sudo apt update
sudo apt install -y wget gnupg2 software-properties-common
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/rocm-archive-keyring.gpg
echo 'deb [signed-by=/usr/share/keyrings/rocm-archive-keyring.gpg] https://repo.radeon.com/rocm/apt/5.7/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

echo "📦 Installing ROCm runtime and tools..."
sudo apt update
sudo apt install -y rocm-hip-runtime rocm-hip-libraries rocm-smi rocm-device-libs rocm-utils hip-dev

echo "👥 Adding user to video and render groups..."
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

echo "🔁 You MUST reboot before continuing!"
echo "🔄 After reboot, run: /opt/rocm/bin/rocminfo"

