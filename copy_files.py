import os
import shutil


def copy_file_multiple_times(original_file, output_dir, n):
    """Copies a single file `n` times."""
    # Check if the original file exists
    if not os.path.isfile(original_file):
        print(f"Error: The file '{original_file}' does not exist.")
        return

    # Get the base name of the original file (without the path)
    base_name = os.path.basename(original_file)

    # Determine the file extension (if any)
    name, extension = os.path.splitext(base_name)

    # Copy the file n times
    for i in range(1, n + 1):
        # Construct the new filename
        new_filename = os.path.join(output_dir, f"{name}_copy{i}{extension}")

        # Copy the file
        shutil.copy2(original_file, new_filename)


def copy_all_files_in_directory(directory, output_dir, n):
    """Copies all files in a directory `n` times."""
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # List all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    # Process each file in the directory
    for file in files:
        original_file = os.path.join(directory, file)
        copy_file_multiple_times(original_file, output_dir, n)


def delete_all_from_directory(directory):
    """Deletes all files and subdirectories from the specified directory."""
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Iterate over all contents in the directory
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            # Check if it's a file or directory and delete accordingly
            if os.path.isfile(item_path) or os.path.islink(item_path):  # Handles files and symbolic links
                os.unlink(item_path)
                print(f"Deleted file: {item_path}")
        except Exception as e:
            print(f"Failed to delete '{item_path}'. Reason: {e}")


# Example usage:
if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "original_10_files"  # Replace with your directory path
    output_path = "isolated_data"
    n = 1  # Number of copies to create for each file
    delete_all_from_directory(output_path)
    copy_all_files_in_directory(directory_path, output_path, n)
