import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class HDataLoader:
    def __init__(self, n, m, k, n_steps, prob_h=None, prob_x=None):
        """
        Initializes the HDataLoader with the specified parameters.

        Parameters:
        - n (int): Number of features per block.
        - m (int): Number of blocks.
        - k (int): Number of x-y pairs per sequence (used in train_dataloader1).
        - n_steps (int): Number of steps per epoch.
        - prob_h (np.array): Optional custom probability vector for h.
        - prob_x (np.array): Optional custom probability vector for features (x) (used in train_dataloader1 and train_dataloader2).
        """
        self.n = n
        self.m = m
        self.k = k
        self.n_steps = n_steps

        # Generate h and identifying x matrices
        self.h_matrix, self.identifying_x_matrix = self._generate_h()

        # Create probability vectors
        self.prob_h, self.prob_x = self._create_probability_vectors(prob_h, prob_x)

        # Total number of h and features
        self.total_h = self.h_matrix.shape[0]
        self.total_features = self.h_matrix.shape[1]

    def _generate_h(self):
        """
        Generates the h matrix and the identifying x matrix.
        """
        num_h_per_block = 2 ** self.n - 2  # Excluding the all-zero vector
        h_list = []

        # Generate all binary combinations of length n, excluding the all-zero vector
        for i in range(1, 2 ** self.n-1):
            # Convert i to a binary vector of length n
            bin_str = format(i, '0' + str(self.n) + 'b')
            h = [int(bit) for bit in bin_str]
            h_list.append(h)
        block = np.array(h_list)  # Shape: (2^n - 2, n)

        total_h = self.m * num_h_per_block
        total_features = self.n * self.m
        h_matrix = np.zeros((total_h, total_features), dtype=int)
        identifying_x_matrix = np.zeros((total_h, total_features), dtype=int)

        for i in range(self.m):
            # Calculate indices for placing the block in the matrices
            h_start = i * num_h_per_block
            h_end = (i + 1) * num_h_per_block
            feature_start = i * self.n
            feature_end = (i + 1) * self.n

            # Place the block into the h matrix
            h_matrix[h_start:h_end, feature_start:feature_end] = block

            # Fill the identifying x matrix with 1s in the block positions
            identifying_block = np.ones((num_h_per_block, self.n), dtype=int)
            identifying_x_matrix[h_start:h_end, feature_start:feature_end] = identifying_block

        return h_matrix, identifying_x_matrix

    def _create_probability_vectors(self, prob_h, prob_x):
        """
        Creates the probability vectors for h and features (x).

        Parameters:
        - prob_h (np.array): Optional custom probability vector for h.
        - prob_x (np.array): Optional custom probability vector for features (x).

        Returns:
        - prob_h (np.array): Probability vector for h.
        - prob_x (np.array): Probability vector for features (x).
        """
        total_h = self.h_matrix.shape[0]
        total_features = self.h_matrix.shape[1]

        if prob_h is None:
            # Create a default probability vector for h (uniform distribution)
            prob_h = np.ones(total_h) / total_h
        else:
            prob_h = np.array(prob_h)
            prob_h /= prob_h.sum()

        if prob_x is None:
            # Create a default probability vector for features (x) (uniform distribution)
            prob_x = np.ones(total_features) / total_features
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
        Sets custom probability vectors for h and features (x).

        Parameters:
        - prob_h (np.array): Custom probability vector for h.
        - prob_x (np.array): Custom probability vector for features (x).
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
                mask_batch = []

                for idx in indices:
                    h = self.h_matrix[idx]
                    identifying_x = self.identifying_x_matrix[idx]
                    #h_batch.append(h)

                    # Generate the located (x, y) pairs
                    located_feature_indices = np.where(identifying_x == 1)[0]
                    located_xs = []
                    located_ys = []
                    mask_sequence_base = []
                    for feature_index in located_feature_indices:
                        y = h[feature_index]
                        x = np.zeros(self.total_features, dtype=int)
                        x[feature_index] = 1
                        located_xs.append(x)
                        located_ys.append(y)
                        mask_sequence_base.append(0)  # Mask 0 for located xs

                    #repeat prefix
                    if prefix_repeat != None:
                        located_xs = located_xs*prefix_repeat
                        located_ys = located_ys*prefix_repeat
                        mask_sequence_base = mask_sequence_base*prefix_repeat
                        
                    # For each additional feature, create a sample by appending the additional (x, y) pair
                    additional_feature_indices = np.arange(self.total_features)
                    for feature_index in additional_feature_indices:
                        # Copy the located xs, ys, and mask
                        x_seq = located_xs.copy()
                        y_seq = located_ys.copy()
                        mask_seq = mask_sequence_base.copy()

                        # Append the additional (x, y) pair
                        y = h[feature_index]
                        x = np.zeros(self.total_features, dtype=int)
                        x[feature_index] = 1

                        x_seq.append(x)
                        y_seq.append(y)
                        mask_seq.append(1)  # Mask 1 for additional x

                        # Add the sequence to the batch
                        x_batch.append(x_seq)
                        y_batch.append(y_seq)
                        h_batch.append(h)
                        mask_batch.append(mask_seq)

                # Convert sequences to tensors
                seq_len = len(x_batch[0])  # All sequences have the same length
                x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)
                y_batch = torch.tensor(np.array(y_batch), dtype=torch.float32)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.float32)
                mask_batch = torch.tensor(np.array(mask_batch), dtype=torch.float32)

                return x_batch, y_batch, h_batch, mask_batch

        elif dataloader_type == 'train1':
            # Original data loader (train_dataloader1)
            def collate_fn(indices):
                x_batch = []
                y_batch = []
                h_batch = []
                for idx in indices:
                    # Get the h
                    h = self.h_matrix[idx]
                    h_batch.append(h)  # Collect h for the batch

                    # Sample k feature indices according to prob_x with replacement
                    feature_indices = np.random.choice(
                        self.total_features, size=self.k, replace=True, p=self.prob_x
                    )

                    # Initialize sequences for x and y
                    x_sequence = []
                    y_sequence = []

                    for feature_index in feature_indices:
                        # Get y as the value in the h_matrix at (idx, feature_index)
                        y = h[feature_index]

                        # Convert x to one-hot vector
                        x = np.zeros(self.total_features, dtype=int)
                        x[feature_index] = 1

                        # Append to the sequence
                        x_sequence.append(x)
                        y_sequence.append(y)

                    # Append sequences to the batch
                    x_batch.append(x_sequence)
                    y_batch.append(y_sequence)

                # Convert batches to tensors
                x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)  # Shape: (batch_size, k, total_features)
                y_batch = torch.tensor(np.array(y_batch), dtype=torch.float32)  # Shape: (batch_size, k)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.float32)  # Shape: (batch_size, total_features)

                return x_batch, y_batch, h_batch, torch.ones(y_batch.shape)

        elif dataloader_type == 'train2':
            # Extended data loader (train_dataloader2)
            def collate_fn(indices):
                x_batch = []
                y_batch = []
                h_batch = []
                mask_batch = []
                for idx in indices:
                    # Get the h and identifying_x for this h
                    h = self.h_matrix[idx]
                    identifying_x = self.identifying_x_matrix[idx]

                    h_batch.append(h)  # Collect h for the batch

                    # Get indices where identifying_x is 1
                    feature_indices = np.where(identifying_x == 1)[0]

                    # Randomize the order
                    np.random.shuffle(feature_indices)

                    # Initialize sequences for x, y, and mask
                    x_sequence = []
                    y_sequence = []
                    mask_sequence = []

                    # For located x's (mask value 0)
                    for feature_index in feature_indices:
                        # Get y as the value in the h_matrix at (idx, feature_index)
                        y = h[feature_index]

                        # Convert x to one-hot vector
                        x = np.zeros(self.total_features, dtype=int)
                        x[feature_index] = 1

                        # Append to the sequence
                        x_sequence.append(x)
                        y_sequence.append(y)
                        mask_sequence.append(0)  # Mask value 0 for located x's   =>  further find this cause issue of local minimum. no mask resolve the issue

                    # Sample one additional x based on prob_x
                    sampled_feature_index = np.random.choice(
                        self.total_features, p=self.prob_x
                    )

                    # Get y for the sampled x
                    y = h[sampled_feature_index]

                    # Convert x to one-hot vector
                    x = np.zeros(self.total_features, dtype=int)
                    x[sampled_feature_index] = 1

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
                    x_seq_padded = x_seq + [np.zeros(self.total_features, dtype=int)] * pad_len
                    y_seq_padded = y_seq + [0] * pad_len
                    mask_seq_padded = mask_seq + [0] * pad_len  # Pad mask with 0s
                    x_batch_padded.append(x_seq_padded)
                    y_batch_padded.append(y_seq_padded)
                    mask_batch_padded.append(mask_seq_padded)

                # Convert batches to tensors
                x_batch = torch.tensor(np.array(x_batch_padded), dtype=torch.float32)  # Shape: (batch_size, max_seq_len, total_features)
                y_batch = torch.tensor(np.array(y_batch_padded), dtype=torch.float32)  # Shape: (batch_size, max_seq_len)
                mask_batch = torch.tensor(np.array(mask_batch_padded), dtype=torch.float32)  # Shape: (batch_size, max_seq_len)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.float32)  # Shape: (batch_size, total_features)

                return x_batch, y_batch, h_batch, mask_batch #torch.ones(y_batch.shape)# 

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

if __name__ == '__main__':
    # Parameters
    n = 2  # Number of features per block
    m = 1  # Number of blocks
    k = 3  # Number of x-y pairs per sequence (used in train_dataloader1)
    n_steps = 5  # Number of steps per epoch
    batch_size = 2  # Number of h per batch
    
    # Initialize the data loader
    data_loader = HDataLoader(n=n, m=m, k=k, n_steps=n_steps)
    
    # Optionally, set custom probability vectors
    '''
    prob_h = np.ones(data_loader.total_h)
    prob_h[0] = 10  # Assign higher probability to the first h
    prob_h /= prob_h.sum()
    
    prob_x = np.ones(data_loader.total_features)
    prob_x[0] = 5  # Assign higher probability to the first feature
    prob_x /= prob_x.sum()
    
    data_loader.set_probability_vectors(prob_h=prob_h, prob_x=prob_x)
    '''
    # Get train_dataloader2
    dataloader = data_loader.get_pytorch_dataloader(batch_size=batch_size, dataloader_type='train2')
    
    # Iterate through train_dataloader2
    for x_batch, y_batch, h_batch, mask_batch in dataloader:
        print("X batch shape:", x_batch.shape)     # Shape: (batch_size, max_seq_len, total_features)
        print("Y batch shape:", y_batch.shape)     # Shape: (batch_size, max_seq_len)
        print("Mask batch shape:", mask_batch.shape)  # Shape: (batch_size, max_seq_len)
        print("H batch shape:", h_batch.shape)     # Shape: (batch_size, total_features)
        print("X batch:")
        print(x_batch)
        print("Y batch:")
        print(y_batch)
        print("Mask batch:")
        print(mask_batch)
        print("H batch:")
        print(h_batch)
        print("-" * 40)
