#!/bin/bash

# Create directories if they don't exist
trainDir="data/training_data"
testDir="data/test_data"
mkdir -p $trainDir $testDir

# Function to download, unzip, and clean up
download_and_extract() {
    local url="$1"
    local dest_dir="$2"

    local file_name=$(basename "$url")
    local zip_path="$dest_dir/$file_name"

    # Download if the file does not exist
    if [ ! -f "$zip_path" ]; then
        wget -q --show-progress -P "$dest_dir" "$url"
    fi

    # Unzip if the file is present and not already extracted
    if [ -f "$zip_path" ] && [ ! -d "${zip_path%.zip}" ]; then
        unzip -q "$zip_path" -d "$dest_dir"
        rm "$zip_path"  # Delete the zip file after extraction
    fi
}

# Download, unzip, and clean up training data
echo "Downloading and extracting training data..."
for i in $(seq 1 3); do # 1 to 17
	url="https://xeek-public-287031953319-eb80.s3.amazonaws.com/image-impeccable/image-impeccable-train-data-part${i}.zip"
    download_and_extract "$url" $trainDir
done

# Download, unzip, and clean up test data
echo "Downloading and extracting test data..."
test_url="https://xeek-public-287031953319-eb80.s3.amazonaws.com/image-impeccable/image-impeccable-test-data.zip"
download_and_extract "$test_url" $testDir

echo "Download, extraction, and cleanup complete!"
