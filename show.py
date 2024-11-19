def clean_edge_list(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Ignore comment lines and empty lines
            if line.startswith(("#", "%")) or not line.strip():
                continue
            
            # Ensure the line has two entries separated by space
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Ensure both parts can be converted to integers (if numeric identifiers are expected)
                    node1, node2 = parts[:2]
                    int(node1)  # Check if numeric, otherwise remove these lines or handle as needed
                    int(node2)
                    outfile.write(f"{node1} {node2}\n")
                except ValueError:
                    continue  # Skip lines with invalid node identifiers

clean_edge_list("./datasets/other/arenas-pgp.edges", "p2p-Gnutella04_cleaned.txt")
