#!/bin/bash

# Fix all splited zip files of EyePACS dataset from Kaggle.
# Also unzips and copy to the new directory all train and test labels, with gradability labels

# Assumes that the dataset resides in ./downloaded_datasets/download_eyePACs_Kaggle/downloads

eyepacs_dir="./data/downloaded_datasets/download_eyePACs_Kaggle/downloads"
new_dir="./data/original_datasets/eyepacs"

# Check if zip is installed.
dpkg -l | grep zip
if [ $? -gt 0 ]; then
    echo "Please install zip: apt-get/yum install zip" >&2
    exit 1
fi

# Check if p7zip is installed.
dpkg -l | grep p7zip-full
if [ $? -gt 0 ]; then
    echo "Please install p7zip-full: apt-get/yum install p7zip-full" >&2
    exit 1
fi

##################################

# All splitted zip files are corrupted, it's neccesary fix them
echo "Fixing train dataset ..."

# Join all splitted zip files (of train dataset) with 'cat'
cat "$eyepacs_dir/train.zip.*" >"$eyepacs_dir/train_joined.zip"

# Remove original zip files
rm "$eyepacs_dir/train.zip.*"

# Fix new joined zip file with 'zip -FF' and save it as 'train_ok.zip'
zip -FF "$eyepacs_dir/train_joined.zip" --out "$eyepacs_dir/train_ok.zip"

# Remove previous joined zip file
rm "$eyepacs_dir/train_joined.zip"

# Unzip new joined zip file to new splitted zip files 
7z e "$eyepacs_dir/train_ok.zip" -o"$eyepacs_dir/."

# Remove joined zip file
rm "$eyepacs_dir/train_ok.zip"

# echo "Unzipping train dataset ..."
# # Unzip new splitted zip files and store their content in the new directory
# THIS DO NOT WORK --> UNZIP MANUALLY
# 7z x "$eyepacs_dir/train.zip.001" -o"$new_dir/images/." || exit 1

###################################

# All splitted zip files are corrupted, it's neccesary fix them
echo "Fixing test dataset ..."

# Join all splitted zip files (of test dataset) with 'cat'
cat $eyepacs_dir/test.zip.* >"$eyepacs_dir/test_joined.zip"

# Remove original zip files
rm $eyepacs_dir/test.zip.*

# Fix new joined zip file with 'zip -FF' and save it as 'train_ok.zip'
zip -FF "$eyepacs_dir/test_joined.zip" --out "$eyepacs_dir/test_ok.zip"

# Remove previous joined zip file
rm "$eyepacs_dir/test_joined.zip"

# Unzip new joined zip file to new splitted zip files 
7z e "$eyepacs_dir/test_ok.zip" -o"$eyepacs_dir/."

# Remove joined zip file
rm "$eyepacs_dir/test_ok.zip"

# echo "Unzipping test dataset ..."
# # Unzip new splitted zip files and store their content in the new directory
# THIS DO NOT WORK --> UNZIP MANUALLY
# 7z x "$eyepacs_dir/test.zip.001" -o"$new_dir/images/." || exit 1

##################################

echo "Unzipping labels for DR ..."
7z e "$eyepacs_dir/trainLabels.csv.zip" -o"$new_dir/." || exit 1
7z e "$eyepacs_dir/testLabels.csv.zip" -o"$new_dir/." || exit 1
echo "Done"

echo "Coying gradability grades ..."
cp "$eyepacs_dir/eyepacs_gradability_grades.csv" "$new_dir/."
echo "Done"