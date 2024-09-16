import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class HDataLoader:
    def __init__(self, n, m, k, n_steps, prob_h=None, prob_x=None):
        """
        Initializes the HDataLoader with the specified parameters.

        Parameters:
        - n (int): Number of features per block.
        - m (float): subsample m hypothesis from all 2^n possibility
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
        num_all_feature = self.n
        num_all_h = 2 ** self.n  # In Excluding the all-zero and all-one vector
        h_list = []

        # Generate all binary combinations of length n
        for i in range(0, 2 ** self.n):
            # Convert i to a binary vector of length n
            bin_str = format(i, '0' + str(self.n) + 'b')
            h = [int(bit) for bit in bin_str]
            h_list.append(h)
        h_matrix = np.array(h_list)  # Shape: (2^n, n)

        if self.m == (2 ** self.n):
            identifying_x_matrix = np.ones((num_all_h, num_all_feature), dtype=int)
        if self.m < (2 ** self.n):
            # calculate identifying_x_matrix
            indices = np.random.choice(2 ** self.n, size=self.m, replace=False)
            h_matrix = h_matrix[indices, :]
            identifying_x_matrix = self._calculate_identifying_x(h_matrix)
        return h_matrix, identifying_x_matrix
    
    def _calculate_identifying_x(self, h_matrix):
        """
        Calculate the identifying_x_matrix using a greedy approach where for each hypothesis,
        the feature that distinguishes it from the most other hypotheses is selected at each step.

        Parameters:
        - h_matrix (np.array): The matrix of hypothesis functions (2^n rows, n columns).

        Returns:
        - identifying_x_matrix (np.array): A binary matrix where 1 indicates that 
          the feature (column) is necessary for distinguishing that hypothesis (row).
        """
        num_h, num_features = h_matrix.shape
        identifying_x_matrix = np.zeros_like(h_matrix, dtype=int)  # Start with a zero matrix

        # For each hypothesis (row in h_matrix)
        for i in range(num_h):
            remaining_hypotheses = set(range(num_h))  # Set of all hypotheses
            remaining_hypotheses.remove(i)  # Remove the current hypothesis from the set

            # Track the features chosen for hypothesis i
            chosen_features = set()

            # Continue until all other hypotheses are distinguished
            while remaining_hypotheses:
                best_feature = None
                max_distinguished = 0

                # Try every feature and see which one distinguishes the most remaining hypotheses
                for feature in range(num_features):
                    if feature in chosen_features:
                        continue  # Skip features already chosen

                    # Count how many remaining hypotheses would be distinguished by this feature
                    distinguished = {j for j in remaining_hypotheses if h_matrix[i, feature] != h_matrix[j, feature]}

                    if len(distinguished) > max_distinguished:
                        max_distinguished = len(distinguished)
                        best_feature = feature

                if best_feature is not None:
                    # Add the best feature to the chosen set and mark it in identifying_x_matrix
                    chosen_features.add(best_feature)
                    identifying_x_matrix[i, best_feature] = 1

                    # Remove distinguished hypotheses from the remaining set
                    distinguished = {j for j in remaining_hypotheses if h_matrix[i, best_feature] != h_matrix[j, best_feature]}
                    remaining_hypotheses -= distinguished

        return identifying_x_matrix

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
                i_batch = []
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
                        i_batch.append(identifying_x)
                        mask_batch.append(mask_seq)

                # Convert sequences to tensors
                seq_len = len(x_batch[0])  # All sequences have the same length
                x_batch = torch.tensor(np.array(x_batch), dtype=torch.float32)
                y_batch = torch.tensor(np.array(y_batch), dtype=torch.float32)
                h_batch = torch.tensor(np.array(h_batch), dtype=torch.float32)
                i_batch = torch.tensor(np.array(i_batch), dtype=torch.float32)
                mask_batch = torch.tensor(np.array(mask_batch), dtype=torch.float32)

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
                    i_batch.append(identifying_x)  # Collect h for the batch

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
                i_batch = torch.tensor(np.array(i_batch), dtype=torch.float32)  # Shape: (batch_size, total_features)

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

                return x_batch, y_batch, h_batch, i_batch, mask_batch #torch.ones(y_batch.shape)# 

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
    n = 3  # Number of features per block
    m = 4  # Number of subsample
    k = 3  # Number of x-y pairs per sequence (used in train_dataloader1)
    n_steps = 5  # Number of steps per epoch
    batch_size = 2  # Number of h per batch
    
    # Initialize the data loader
    data_loader = HDataLoader(n=n, m=m, k=k, n_steps=n_steps)
    
    print(data_loader.h_matrix)
    print(data_loader.identifying_x_matrix)
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
    print("-" * 40)
    for x_batch, y_batch, h_batch, i_batch, mask_batch in dataloader:
        # print("X batch shape:", x_batch.shape)     # Shape: (batch_size, max_seq_len, total_features)
        # print("Y batch shape:", y_batch.shape)     # Shape: (batch_size, max_seq_len)
        # print("Mask batch shape:", mask_batch.shape)  # Shape: (batch_size, max_seq_len)
        # print("H batch shape:", h_batch.shape)     # Shape: (batch_size, total_features)
        print("H batch:")
        print(h_batch)
        print("I batch:")
        print(i_batch)
        print("X batch:")
        print(x_batch)
        print("Y batch:")
        print(y_batch)
        print("Mask batch:")
        print(mask_batch)
        print("-" * 40)
