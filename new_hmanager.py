import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import itertools
import math

class HypothesisManager:
    def __init__(self, mode, n, m, random_seed, k, n_steps, prob_h=None, prob_x=None):
        """
        Initializes the PermutationDataLoader with the specified parameters.

        Parameters:
        - mode (str): 'permutation' or 'binary'
        - n (int): Number of elements in the permutations.
        - m (int): Number of permutations to sample from all n! permutations.
        - random_seed (int): Random seed for reproducibility.
        - k (int): Number of x-y pairs per sequence.
        - n_steps (int): Number of steps per epoch.
        - prob_h (np.array): Optional custom probability vector for h.
        - prob_x (np.array): Optional custom probability vector for positions (x).
        """
        self.mode = mode
        self.n = n
        self.m = m
        self.random_seed = random_seed
        self.k = k
        self.n_steps = n_steps

        # Generate h and identifying x matrices
        self.h_matrix, self.identifying_x_matrix, self.num_all_h, self.tokens = self._generate_h(mode, n)

        # Create probability vectors
        self.prob_h, self.prob_x = self._create_probability_vectors(prob_h, prob_x)

        # Total number of h and features (positions)
        self.total_h = len(self.h_matrix)
        self.total_positions = n  # Positions from 0 to n-1

    def _generate_h(self, mode, n):
        """
        Generates the h matrix (list of permutations or binary combinations) and the identifying x matrix.
        """
        if mode == 'permutation':
            # Generate all permutations
            all_perms = list(itertools.permutations(range(0, n)))
            num_all_h = len(all_perms) # Should be n!
            if num_all_h != math.factorial(n):
                raise Exception('wrong num_all_h')

            if self.m >= num_all_h:
                h_matrix = all_perms
            else:
                np.random.seed(self.random_seed)
                indices = np.random.choice(num_all_h, size=self.m, replace=False)
                h_matrix = [all_perms[i] for i in indices]

            # Convert h_matrix to numpy array
            h_matrix = np.array(h_matrix)  # Shape: (m, n)

            # Calculate identifying_x_matrix
            identifying_x_matrix = self._calculate_identifying_x(h_matrix)

            tokens = n + 1  # Since labels range from 0 to n-1, maximum label is n-1, padding value is n, so tokens = n+1.

        elif mode == 'binary':
            # Generate all combinations of n positions with labels n+1 and n+2
            labels = [n+1, n+2]
            all_combinations = list(itertools.product(labels, repeat=n))
            num_all_h = len(all_combinations)  # Should be 2^n
            if num_all_h != (2**n):
                raise Exception('wrong num_all_h')
            
            if self.m >= num_all_h:
                h_matrix = all_combinations
            else:
                np.random.seed(self.random_seed)
                indices = np.random.choice(num_all_h, size=self.m, replace=False)
                h_matrix = [all_combinations[i] for i in indices]

            # Convert h_matrix to numpy array
            h_matrix = np.array(h_matrix)  # Shape: (m, n)

            # Calculate identifying_x_matrix
            identifying_x_matrix = self._calculate_identifying_x(h_matrix)

            tokens = n + 3  # Since labels are n+1 and n+2, maximum label is n+2, so tokens = n+3.

        else:
            raise ValueError("Invalid mode. Must be 'permutation' or 'binary'.")

        return h_matrix, identifying_x_matrix, num_all_h, tokens

    def _calculate_identifying_x(self, h_matrix):
        """
        Calculate the identifying_x_matrix using a greedy approach where for each hypothesis,
        the position that distinguishes it from the most other hypotheses is selected at each step.

        Parameters:
        - h_matrix (np.array): The matrix of permutations (rows: hypotheses, columns: positions).

        Returns:
        - identifying_x_matrix (np.array): A binary matrix where 1 indicates that 
          the position (column) is necessary for distinguishing that hypothesis (row).
        """
        num_h, n = h_matrix.shape
        identifying_x_matrix = np.zeros_like(h_matrix, dtype=int)  # Start with a zero matrix

        # For each hypothesis (row in h_matrix)
        for i in range(num_h):
            remaining_hypotheses = set(range(num_h))  # Set of all hypotheses
            remaining_hypotheses.remove(i)  # Remove the current hypothesis from the set

            # Track the positions chosen for hypothesis i
            chosen_positions = set()

            # Continue until all other hypotheses are distinguished
            while remaining_hypotheses:
                best_position = None
                max_distinguished = 0

                # Try every position and see which one distinguishes the most remaining hypotheses
                for position in range(n):
                    if position in chosen_positions:
                        continue  # Skip positions already chosen

                    # Count how many remaining hypotheses would be distinguished by this position
                    distinguished = {j for j in remaining_hypotheses if h_matrix[i, position] != h_matrix[j, position]}

                    if len(distinguished) > max_distinguished:
                        max_distinguished = len(distinguished)
                        best_position = position

                if best_position is not None:
                    # Add the best position to the chosen set and mark it in identifying_x_matrix
                    chosen_positions.add(best_position)
                    identifying_x_matrix[i, best_position] = 1

                    # Remove distinguished hypotheses from the remaining set
                    distinguished = {j for j in remaining_hypotheses if h_matrix[i, best_position] != h_matrix[j, best_position]}
                    remaining_hypotheses -= distinguished
                else:
                    raise Exception('Something unexpected happens for _calculate_identifying_x')

        return identifying_x_matrix

    def _create_probability_vectors(self, prob_h, prob_x):
        """
        Creates the probability vectors for h and positions (x).

        Parameters:
        - prob_h (np.array): Optional custom probability vector for h.
        - prob_x (np.array): Optional custom probability vector for positions (x).

        Returns:
        - prob_h (np.array): Probability vector for h.
        - prob_x (np.array): Probability vector for positions (x).
        """
        total_h = self.h_matrix.shape[0]
        total_positions = self.n  # Positions from 0 to n-1

        if prob_h is None:
            # Create a default probability vector for h (uniform distribution)
            prob_h = np.ones(total_h) / total_h
        else:
            prob_h = np.array(prob_h)
            prob_h /= prob_h.sum()

        if prob_x is None:
            # Create a default probability vector for positions (x) (uniform distribution)
            prob_x = np.ones(total_positions) / total_positions
        else:
            prob_x = np.array(prob_x)
            prob_x /= prob_x.sum()

        return prob_h, prob_x

    def get_h_matrix(self):
        """
        Returns the h matrix.

        Returns:
        - h_matrix (np.array): The h matrix.
        """
        return self.h_matrix

    def get_identifying_x_matrix(self):
        """
        Returns the identifying x matrix.

        Returns:
        - identifying_x_matrix (np.array): The identifying x matrix.
        """
        return self.identifying_x_matrix

    def set_probability_vectors(self, prob_h=None, prob_x=None):
        """
        Sets custom probability vectors for h and positions (x).

        Parameters:
        - prob_h (np.array): Custom probability vector for h.
        - prob_x (np.array): Custom probability vector for positions (x).
        """
        self.prob_h, self.prob_x = self._create_probability_vectors(prob_h, prob_x)

    def get_pytorch_dataloader(self, batch_size=1, dataloader_type='train1', prefix_repeat=None):
        """
        Creates a PyTorch DataLoader using the h and probabilities.

        Parameters:
        - batch_size (int): Number of h per batch. For the test data loader, `batch_size` is set to `1` internally, and each batch contains all samples for one hypothesis.
        - dataloader_type (str): Type of data loader ('train1', 'train2', or 'test').

        Returns:
        - data_loader (DataLoader): A PyTorch DataLoader.
        """
        # Define the custom Dataset
        class HDataset(Dataset):
            def __init__(dataset_self):
                dataset_self.h_matrix = self.h_matrix
                dataset_self.identifying_x_matrix = self.identifying_x_matrix
                dataset_self.total_h = self.total_h

            def __len__(dataset_self):
                return dataset_self.total_h

            def __getitem__(dataset_self, idx):
                # Return the index; we'll retrieve the h in collate_fn
                return idx

        # Create the dataset
        dataset = HDataset()

        if dataloader_type == 'test':
            sampler = None
            shuffle = False  # Do not shuffle test data
            batch_size = 1  # Process one h at a time
        else:
            # For training data loaders, use WeightedRandomSampler
            num_samples = self.n_steps * batch_size
            sampler = WeightedRandomSampler(
                weights=self.prob_h,
                num_samples=num_samples,
                replacement=True
            )
            shuffle = False  # When using a sampler, shuffle must be False

        # Define the collate function based on dataloader_type
        if dataloader_type == 'test':
            def collate_fn(indices):
                x_batch = []
                y_batch = []
                h_batch = []
                i_batch = []
                mask_batch = []

                for idx in indices:
                    h = self.h_matrix[idx]
                    identifying_x = self.identifying_x_matrix[idx]

                    # Generate the located (x, y) pairs
                    located_position_indices = np.where(identifying_x == 1)[0]
                    located_xs = []
                    located_ys = []
                    mask_sequence_base = []
                    for position_index in located_position_indices:
                        y = h[position_index]
                        x = position_index  # x is the position index

                        located_xs.append(x)
                        located_ys.append(y)
                        mask_sequence_base.append(0)  # Mask 0 for located xs

                    # Repeat prefix if needed
                    if prefix_repeat is not None:
                        located_xs = located_xs * prefix_repeat
                        located_ys = located_ys * prefix_repeat
                        mask_sequence_base = mask_sequence_base * prefix_repeat

                    # For each additional position, create a sample by appending the additional (x, y) pair
                    additional_position_indices = np.arange(self.n)
                    for position_index in additional_position_indices:
                        # Copy the located xs, ys, and mask
                        x_seq = located_xs.copy()
                        y_seq = located_ys.copy()
                        mask_seq = mask_sequence_base.copy()

                        # Append the additional (x, y) pair
                        y = h[position_index]
                        x = position_index  # x is the position index

                        x_seq.append(x)
                        y_seq.append(y)
                        mask_seq.append(1)  # Mask 1 for additional x

                        # Add the sequence to the batch
                        x_batch.append(x_seq)
                        y_batch.append(y_seq)
                        h_batch.append(h)
                        i_batch.append(identifying_x)
                        mask_batch.append(mask_seq)

                # Convert sequences to tensors
                max_seq_len = max(len(seq) for seq in x_batch)

                # Pad sequences to have the same length
                x_batch_padded = []
                y_batch_padded = []
                mask_batch_padded = []
                for x_seq, y_seq, mask_seq in zip(x_batch, y_batch, mask_batch):
                    # Pad x_seq, y_seq, and mask_seq to max_seq_len
                    pad_len = max_seq_len - len(x_seq)
                    x_seq_padded = x_seq + [self.n] * pad_len  # Pad x_seq with n
                    y_seq_padded = y_seq + [self.n] * pad_len  # Pad y_seq with n
                    mask_seq_padded = mask_seq + [0] * pad_len  # Pad mask with 0s
                    x_batch_padded.append(x_seq_padded)
                    y_batch_padded.append(y_seq_padded)
                    mask_batch_padded.append(mask_seq_padded)

                x_batch = torch.tensor(np.array(x_batch_padded), dtype=torch.long)
                y_batch = torch.tensor(np.array(y_batch_padded), dtype=torch.long)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.long)
                i_batch = torch.tensor(np.array(i_batch), dtype=torch.long)
                mask_batch = torch.tensor(np.array(mask_batch_padded), dtype=torch.float32)

                return x_batch, y_batch, h_batch, i_batch, mask_batch

        elif dataloader_type == 'train1':
            # Original data loader (train_dataloader1)
            def collate_fn(indices):
                x_batch = []
                y_batch = []
                h_batch = []
                i_batch = []
                for idx in indices:
                    # Get the h
                    h = self.h_matrix[idx]
                    identifying_x = self.identifying_x_matrix[idx]
                    h_batch.append(h)  # Collect h for the batch
                    i_batch.append(identifying_x)  # Collect identifying_x for the batch

                    # Sample k position indices according to prob_x with replacement
                    position_indices = np.random.choice(
                        self.n, size=self.k, replace=True, p=self.prob_x
                    )

                    # Initialize sequences for x and y
                    x_sequence = []
                    y_sequence = []

                    for position_index in position_indices:
                        # Get y as the value in h at position_index
                        y = h[position_index]

                        # x is the position index
                        x = position_index

                        # Append to the sequence
                        x_sequence.append(x)
                        y_sequence.append(y)

                    # Append sequences to the batch
                    x_batch.append(x_sequence)
                    y_batch.append(y_sequence)

                # Convert batches to tensors
                x_batch = torch.tensor(np.array(x_batch), dtype=torch.long)  # Shape: (batch_size, k)
                y_batch = torch.tensor(np.array(y_batch), dtype=torch.long)  # Shape: (batch_size, k)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.long)  # Shape: (batch_size, n)
                i_batch = torch.tensor(np.array(i_batch), dtype=torch.long)  # Shape: (batch_size, n)

                return x_batch, y_batch, h_batch, i_batch, torch.ones(y_batch.shape)

        elif dataloader_type == 'train2':
            # Extended data loader (train_dataloader2)
            def collate_fn(indices):
                x_batch = []
                y_batch = []
                h_batch = []
                i_batch = []
                mask_batch = []
                for idx in indices:
                    # Get the h and identifying_x for this h
                    h = self.h_matrix[idx]
                    identifying_x = self.identifying_x_matrix[idx]

                    h_batch.append(h)  # Collect h for the batch
                    i_batch.append(identifying_x)

                    # Get indices where identifying_x is 1
                    position_indices = np.where(identifying_x == 1)[0]

                    # Randomize the order
                    np.random.shuffle(position_indices)

                    # Initialize sequences for x, y, and mask
                    x_sequence = []
                    y_sequence = []
                    mask_sequence = []

                    # For located x's (mask value 0)
                    for position_index in position_indices:
                        # Get y as the value in h at position_index
                        y = h[position_index]

                        # x is the position index
                        x = position_index

                        # Append to the sequence
                        x_sequence.append(x)
                        y_sequence.append(y)
                        mask_sequence.append(0)  # Mask value 0 for located x's

                    # Sample one additional x based on prob_x
                    sampled_position_index = np.random.choice(
                        self.n, p=self.prob_x
                    )

                    # Get y for the sampled x
                    y = h[sampled_position_index]

                    # x is the position index
                    x = sampled_position_index

                    # Append to the sequence
                    x_sequence.append(x)
                    y_sequence.append(y)
                    mask_sequence.append(1)  # Mask value 1 for new sampled x

                    # Append sequences to the batch
                    x_batch.append(x_sequence)
                    y_batch.append(y_sequence)
                    mask_batch.append(mask_sequence)

                # Find the maximum sequence length in the batch
                max_seq_len = max(len(seq) for seq in x_batch)

                # Pad sequences to have the same length
                x_batch_padded = []
                y_batch_padded = []
                mask_batch_padded = []
                for x_seq, y_seq, mask_seq in zip(x_batch, y_batch, mask_batch):
                    # Pad x_seq, y_seq, and mask_seq to max_seq_len
                    pad_len = max_seq_len - len(x_seq)
                    x_seq_padded = x_seq + [self.n] * pad_len  # Pad x_seq with n
                    y_seq_padded = y_seq + [self.n] * pad_len  # Pad y_seq with n
                    mask_seq_padded = mask_seq + [0] * pad_len  # Pad mask with 0s
                    x_batch_padded.append(x_seq_padded)
                    y_batch_padded.append(y_seq_padded)
                    mask_batch_padded.append(mask_seq_padded)

                # Convert batches to tensors
                x_batch = torch.tensor(np.array(x_batch_padded), dtype=torch.long)  # Shape: (batch_size, max_seq_len)
                y_batch = torch.tensor(np.array(y_batch_padded), dtype=torch.long)  # Shape: (batch_size, max_seq_len)
                mask_batch = torch.tensor(np.array(mask_batch_padded), dtype=torch.float32)  # Shape: (batch_size, max_seq_len)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.long)  # Shape: (batch_size, n)
                i_batch = torch.tensor(np.array(i_batch), dtype=torch.long)  # Shape: (batch_size, n)

                return x_batch, y_batch, h_batch, i_batch, mask_batch

        else:
            raise ValueError("Invalid dataloader_type. Must be 'train1', 'train2', or 'test'.")

        # Create the DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,  # Must be False when sampler is provided
            collate_fn=collate_fn,
            num_workers=0
        )

        return data_loader

from utils.nano_gpt import label2onehot
def combine2(xs_b, ys_b, dim):
    """Interleaves the x's and the y's into a single sequence."""
    bsize, points = xs_b.shape
    '''
    ys_b_wide = torch.cat(
        (
            ys_b.view(bsize, points, 1),
            torch.zeros(bsize, points, dim - 1, device=ys_b.device),
        ),
        axis=2,
    )
    '''
    # zs = torch.stack((torch.cat([xs_b,                                               # x
    #                              torch.zeros([bsize, points, 1], device=ys_b.device) # y
    #                              ], dim=2), # x
    #                   torch.cat([torch.zeros_like(xs_b, device=ys_b.device),         # x
    #                              ys_b.view(bsize, points, 1)                         # y
    #                              ], dim=2)
    #                   ), dim=2)
    #print('forward')
    #print(xs_b.shape)
    #print(ys_b.shape)
    zs = torch.stack((label2onehot(xs_b, dim), # x
                      label2onehot(ys_b, dim), # y
                     ), dim=2)

    zs = zs.view(bsize, 2 * points, dim)
    #print(zs.shape)
    return zs

if __name__ == '__main__':
    # Parameters
    mode = 'binary'
    n = 4  # Number of elements in the permutations
    m = 8  # Number of permutations to sample (6 for all permutations of 3 elements)
    random_seed = 1
    k = 3  # Number of x-y pairs per sequence (used in train_dataloader1)
    n_steps = 32  # Number of steps per epoch
    batch_size = 2  # Number of h per batch

    # Initialize the data loader
    hmanager = HypothesisManager(mode=mode, n=n, m=m, random_seed=random_seed, k=k, n_steps=n_steps)

    print('H sampling')
    print(hmanager.m, hmanager.num_all_h)
    print("Hypotheses (h_matrix):")
    print(hmanager.h_matrix)
    print("Identifying positions (identifying_x_matrix):")
    print(hmanager.identifying_x_matrix)

    # Optionally, set custom probability vectors
    '''
    prob_h = np.ones(data_loader.total_h)
    prob_h[0] = 10  # Assign higher probability to the first h
    prob_h /= prob_h.sum()

    prob_x = np.ones(data_loader.total_positions)
    prob_x[0] = 5  # Assign higher probability to the first position
    prob_x /= prob_x.sum()

    data_loader.set_probability_vectors(prob_h=prob_h, prob_x=prob_x)
    '''

    # Get train_dataloader2
    dataloader = hmanager.get_pytorch_dataloader(batch_size=batch_size, dataloader_type='train2', prefix_repeat=1)

    # Iterate through train_dataloader2
    print("-" * 40)
    for x_batch, y_batch, h_batch, i_batch, mask_batch in dataloader:
        print("X batch shape:", x_batch.shape)     # Shape: (batch_size, max_seq_len)
        print("Y batch shape:", y_batch.shape)     # Shape: (batch_size, max_seq_len)
        print("Mask batch shape:", mask_batch.shape)  # Shape: (batch_size, max_seq_len)
        print("H batch shape:", h_batch.shape)     # Shape: (batch_size, n)
        print("H batch:")
        print(h_batch)
        print("I batch:")
        print(i_batch)
        print("X batch:")
        print(x_batch)
        print("Y batch:")
        print(y_batch)
        print("Z batch")
        print(combine2(x_batch, y_batch, hmanager.tokens))
        print("Mask batch:")
        print(mask_batch)
        print("-" * 40)
        
        break  # Remove this line to iterate over the entire dataset