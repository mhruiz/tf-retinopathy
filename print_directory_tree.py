import sys
import os

def print_directory_tree(path="", directory="", depth=0, max_depth=3):

    def generate_level(depth):
        if (depth > 0):
            level = " |   " * (depth - 1) + " |->"
        else:
            level = ""
        return level

    if depth == 0:
        print(os.path.split(sys.path[0])[-1], "/")
        next_level = os.listdir(sys.path[0])
    elif depth <= max_depth:
        print(generate_level(depth), directory, "/")
        next_level = os.listdir(os.path.join(path, directory))
    else:
        return

    depth += 1

    # Check if there is at least one directory inside, if there is no directory, print as list
    if next((x for x in next_level if os.path.isdir(os.path.join(path, directory, x))), None) is None or depth > max_depth:
        inside = len(next_level)
        print(generate_level(depth),", ".join(next_level[:4]),"..." if inside > 4 else "")
    else:
        for item in next_level:
            full_path = os.path.join(path, directory, item)
            if os.path.isdir(full_path):
                print_directory_tree(os.path.join(path, directory), item, depth, max_depth)
            else:
                print(generate_level(depth), item)

print_directory_tree()