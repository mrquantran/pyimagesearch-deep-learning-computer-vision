# copy images from flowers17 dataset to a new directory
import os 
import shutil # copy files

# Define the source and target directories
source_dir = './datasets/flowers17'
target_dir = './datasets/target_flowers17'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Function to copy and rename images
def copy_and_rename_images(source, target):
    for label in os.listdir(source):
        label_dir = os.path.join(source, label)
        if os.path.isdir(label_dir):
            for index, filename in enumerate(os.listdir(label_dir)):
                source_file = os.path.join(label_dir, filename)
                target_file = os.path.join(target, f'{label}_{index + 1}.jpg')
                shutil.copy2(source_file, target_file)
                print(f'Copied and renamed: {source_file} -> {target_file}')


# Copy and rename the images
copy_and_rename_images(source_dir, target_dir)
