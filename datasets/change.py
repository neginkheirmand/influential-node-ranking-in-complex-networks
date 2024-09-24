import os

input_directory = "./"
output_directory = "./"

for dirpath, _, files in os.walk(input_directory):
    try:
        # with open('./arenas-pgp.edge', 'r') as f:
        # with open('./politician_edges.csv', 'r') as f:
        with open('./tvshow_edges.edge', 'r') as f:
            lines = f.readlines()
            corrected_lines = [line.replace(',', ' ').strip() for line in lines]

        with open('./tvshow_edges.edges', 'w') as f:
            f.write('\n'.join(corrected_lines) + '\n')  

    except Exception as e:
        print(f"Error processing ./tvshow_edges.csv: {e}")
