#!/bin/bash
set -e

I2A_S1=https://drive.google.com/drive/folders/1uNQucxKvYcuv8UwWKcDSsUk9K_IzskxY?usp=drive_link

if [ ! -d "$HOME/Documents/I2A_S1" ]; then
    echo "$HOME/Documents/I2A_S1 found. Download the drive folder at:"
    echo "      $I2A_S1"
    echo ""
else
    ln -sfn $HOME/Documents/I2A_S1/TAI/Images/ ./
    ln -sfn $HOME/Documents/I2A_S1/TAI/Images_TP2_EXO6_SR/ ./
fi
