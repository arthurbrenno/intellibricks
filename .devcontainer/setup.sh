#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print section headers
print_section() {
    echo "==== $1 ===="
}

# Install UV package installer
print_section "Installing UV"
pip install --root-user-action=ignore --upgrade uv

# Print success message
print_section "Installation Complete"
