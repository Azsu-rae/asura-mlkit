#!/bin/bash
set -e

TAI_IMAGES=https://drive.google.com/drive/folders/1Q3pt4DxMoC6d35xCjBFh5pw89BAINSUD?usp=sharing

ans=""
if [ ! -d "./TAI" ]; then
    echo "you can find the dataset in"
    echo ""
    echo "          $TAI_IMAGES"
    echo ""
    echo "and then place them in this directory. or,"
    echo "an attempt at downloading them directly can be made using gdown (could fail due to google throttling)."
    read -p "attempt [y/n]? " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        ans=""
    fi

    if [[ -n "$ans" ]]; then
        if ! command -v gdown > /dev/null; then
            echo "gdown is required to fetch from google drive"
            echo "install using \`pipx install gdown\`"
        else
            gdown --folder $TAI_IMAGES
        fi
    else
        echo "skipped."
    fi
else
    echo "./TAI already exists in this directory. aborting."
fi
