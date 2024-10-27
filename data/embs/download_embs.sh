#!/bin/bash

# List of languages
LIST_OF_LANGS=("hi" "en" "fr" "de" "ja" "et" "ru")

# Base URL
BASE_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.LANG.vec"

# Loop over each language
for LANG in "${LIST_OF_LANGS[@]}"; do
  # Construct the URL
  URL=${BASE_URL//LANG/$LANG}
  
  # Extract the filename from the URL
  FILENAME=$(basename $URL)
  
  # Check if the file already exists
  if [ -f "$FILENAME" ]; then
    echo "$FILENAME already exists. Skipping download."
  else
    # Download the file
    wget $URL
  fi
done