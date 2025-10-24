import os

def print_tree(root_dir, prefix="", level=0, max_depth=4):
    if level >= max_depth:
        return

    entries = sorted(os.listdir(root_dir))
    entries_count = len(entries)

    for index, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        connector = "└── " if index == entries_count - 1 else "├── "
        print(prefix + connector + entry)
        
        if os.path.isdir(path):
            extension = "    " if index == entries_count - 1 else "│   "
            print_tree(path, prefix + extension, level + 1, max_depth)

# Use current working directory as root
root_directory = os.getcwd()
print(root_directory)
print_tree(root_directory)
