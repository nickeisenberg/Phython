import os

path_d = '/Users/nickeisenberg/GitRepos/Phython/Notebook/Func_Outputs/'
file_name = 'created_text_file.txt'

# create the file
Func_Outputs = [f for f in os.listdir(path_d)]
if file_name not in Func_Outputs:
    open(path_d + file_name, 'x')

# write to the file
with open(path_d + file_name, 'w') as f:
    f.write('line 1\n')
    f.write('line 2')

with open(path_d + file_name, 'r') as r:
    lines = r.read()

print(lines)









