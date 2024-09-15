import numpy as np

def generate_hypotheses(n, m):
    num_hypotheses_per_block = 2 ** n - 1  # Excluding the all-zero vector
    hypotheses_list = []
    
    # Generate all binary combinations of length n, excluding the all-zero vector
    for i in range(1, 2 ** n):
        # Convert i to a binary vector of length n
        bin_str = format(i, '0' + str(n) + 'b')
        hypothesis = [int(bit) for bit in bin_str]
        hypotheses_list.append(hypothesis)
    block = np.array(hypotheses_list)  # Shape: (2^n - 1, n)

    total_hypotheses = m * num_hypotheses_per_block
    total_features = n * m
    hypotheses_matrix = np.zeros((total_hypotheses, total_features), dtype=int)
    identifying_x_matrix = np.zeros((total_hypotheses, total_features), dtype=int)

    for i in range(m):
        # Calculate indices for placing the block in the matrices
        hypothesis_start = i * num_hypotheses_per_block
        hypothesis_end = (i + 1) * num_hypotheses_per_block
        feature_start = i * n
        feature_end = (i + 1) * n

        # Place the block into the hypotheses matrix
        hypotheses_matrix[hypothesis_start:hypothesis_end, feature_start:feature_end] = block

        # Fill the identifying x matrix with 1s in the block positions
        identifying_block = np.ones((num_hypotheses_per_block, n), dtype=int)
        identifying_x_matrix[hypothesis_start:hypothesis_end, feature_start:feature_end] = identifying_block

    return hypotheses_matrix, identifying_x_matrix

# Example usage:
n = 3  # Number of features per block
m = 3  # Number of blocks
hypotheses_matrix, identifying_x_matrix = generate_hypotheses(n, m)
print("Hypotheses Matrix:")
print(hypotheses_matrix)
print("\nIdentifying X Matrix:")
print(identifying_x_matrix)