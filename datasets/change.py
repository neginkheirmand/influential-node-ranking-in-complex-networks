import os

input_directory = "./"
output_directory = "./"

def edge2edges_file(edge_file, edges_file):
    for dirpath, _, files in os.walk(input_directory):
        try:
            with open(edge_file, 'r') as f:
                lines = f.readlines()
                corrected_lines = [line.replace(',', ' ').strip() for line in lines]

            with open(edges_file, 'w') as f:
                f.write('\n'.join(corrected_lines) + '\n')  

        except Exception as e:
            print(f"Error processing ./tvshow_edges.csv: {e}")


edges_file = './tvshow_edges.edges'
edge_file = './tvshow_edges.edge'
edge2edges_file(edge_file, edges_file)