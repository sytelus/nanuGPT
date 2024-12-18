import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from nanugpt.data.tokenized_data import MemmapDataset, MemmapDataloader

class TestMemmapDataloader(unittest.TestCase):
    def setUp(self):
        # Mocking MemmapDataset
        self.mock_data = np.arange(10)
        self.mock_dataset = MagicMock(spec=MemmapDataset)
        self.mock_dataset.context_length = 3
        self.mock_dataset.data = self.mock_data
        self.mock_dataset.seq_count = len(self.mock_data) - self.mock_dataset.context_length + 1
        self.mock_dataset.seq_len = None
        self.mock_dataset.set_seq_len = lambda seq_len: setattr(self.mock_dataset, 'seq_len', seq_len)
        self.mock_dataset.__len__.return_value = self.mock_dataset.seq_count
        self.mock_dataset.__getitem__.side_effect = lambda idx: MemmapDataset.__getitem__(self.mock_dataset, idx)
        pass

    def test_batch_size_greater_than_dataset_length(self):
        dataloader = MemmapDataloader(self.mock_dataset, batch_size=10, seed=42, shuffle=False)
        self.assertEqual(dataloader.batch_count, 1)
        batches = list(dataloader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 10)
        self.assertTrue(np.array_equal(batches[0][0], [[0, 1, 2],
                                                        [3, 4, 5],
                                                        [6, 7, 8],
                                                        [9, 0, 1],
                                                        [2, 3, 4],
                                                        [5, 6, 7],
                                                        [8, 9, 0],
                                                        [1, 2, 3],
                                                        [4, 5, 6],
                                                        [7, 8, 9]]) )

    def test_batch_size_less_than_dataset_length(self):
        dataloader = MemmapDataloader(self.mock_dataset, batch_size=2, seed=42, shuffle=False)
        self.assertEqual(dataloader.batch_count, 2)
        batches = list(dataloader)
        self.assertEqual(len(batches), 2)
        x, y = batches[0]
        self.assertEqual(x.shape, (2,3))
        self.assertEqual(y.shape, (2,3))

        self.assertTrue(np.array_equal(batches[0][0], [[0, 1, 2],
                                                        [3, 4, 5],]))
        self.assertTrue(np.array_equal(batches[0][1], [[1, 2, 3],
                                                        [4, 5, 6],]))

    def test_batch_size_greater_than_dataset_length_shuffled(self):
        dataloader = MemmapDataloader(self.mock_dataset, batch_size=10, seed=42, shuffle=True)
        self.assertEqual(dataloader.batch_count, 1)
        batches = list(dataloader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0][0].shape[0], 10)
        self.assertTrue(np.array_equal(batches[0][0], [[1, 2, 3],
                                                        [4, 5, 6],
                                                        [7, 8, 9],
                                                        [0, 1, 2],
                                                        [3, 4, 5],
                                                        [6, 7, 8],
                                                        [9, 0, 1],
                                                        [2, 3, 4],
                                                        [5, 6, 7],
                                                        [8, 9, 0]]))
        self.assertTrue(np.array_equal(batches[0][1], [[2, 3, 4],
                                                        [5, 6, 7],
                                                        [8, 9, 0],
                                                        [1, 2, 3],
                                                        [4, 5, 6],
                                                        [7, 8, 9],
                                                        [0, 1, 2],
                                                        [3, 4, 5],
                                                        [6, 7, 8],
                                                        [9, 0, 1]]))

    def test_batch_size_less_than_dataset_length_shuffled(self):
        dataloader = MemmapDataloader(self.mock_dataset, batch_size=2, seed=42, shuffle=True)
        self.assertEqual(dataloader.batch_count, 2)
        batches = list(dataloader)
        self.assertEqual(len(batches), 2)
        x, y = batches[0]
        self.assertEqual(x.shape, (2,3))
        self.assertEqual(y.shape, (2,3))
        x, y = batches[1]
        self.assertEqual(x.shape, (2,3))
        self.assertEqual(y.shape, (2,3))
        self.assertTrue(np.array_equal(batches[0][0],  [[1, 2, 3],
                                                        [4, 5, 6]]))
        self.assertTrue(np.array_equal(batches[1][0], [[2, 3, 4],
                                                        [5, 6, 7]]))


