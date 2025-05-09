#!/bin/bash

echo "=== 🧠 ROCm + AMD GPU Fixer Script ==="

# Step 1: Check Secure Boot
echo -e "\n🔒 Checking Secure Boot status..."
if mokutil --sb-state | grep -qi "enabled"; then
  echo "❌ Secure Boot is ENABLED. ROCm kernel drivers will NOT load."
  echo "➡️ Please reboot into BIOS and disable Secure Boot, then rerun this script."
  exit 1
else
  echo "✅ Secure Boot is disabled."
fi

# Step 2: Detect AMD GPU
echo -e "\n🔍 Detecting AMD GPU on PCI bus..."
lspci | grep -Ei 'vga|3d|display|amd|ati'

# Step 3: Verify amdgpu driver binding
echo -e "\n🔍 Checking if amdgpu driver is bound..."
lspci -nnk | grep -A 3 VGA | grep -i amdgpu || echo "❗ amdgpu kernel driver NOT bound yet"

# Step 4: Patch GRUB for ROCm hybrid issues
echo -e "\n🛠 Patching /etc/default/grub with ROCm-friendly boot flags..."

sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.runpm=0 amdgpu.ppfeaturemask=0xffffffff iommu=soft"/' /etc/default/grub
sudo update-grub

# Step 5: Ensure amdgpu module auto-loads
echo "✅ Ensuring amdgpu module loads at boot..."
echo amdgpu | sudo tee /etc/modules-load.d/amdgpu.conf
sudo update-initramfs -u

# Step 6: Ensure DKMS + headers are installed
echo -e "\n📦 Verifying kernel headers and DKMS..."
sudo apt install -y dkms linux-headers-$(uname -r)

# Step 7: Reinstall AMDGPU with ROCm kernel modules
echo -e "\n♻️ Reinstalling ROCm kernel driver stack (with DKMS)..."
sudo amdgpu-install --usecase=rocm -y

# Step 8: Prompt reboot
echo -e "\n✅ Setup complete. Please reboot for changes to take effect."
read -p "Reboot now? [Y/n] " yn
case $yn in
  [Yy]* ) sudo reboot;;
  * ) echo "❗ Please reboot manually to apply changes.";;
esac
