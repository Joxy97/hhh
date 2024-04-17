import numpy as np
from prettytable import PrettyTable
import onnxruntime
import h5py
import json
from itertools import permutations
import math
import sys
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# Takes in an arbitrary shape tensor and returns the same shape tensor with 1s at the positions of maximum elements
def one_hot_encode(input):
  if torch.any(input != 0):
    max_value = input.view(-1).max()
    max_indices = torch.nonzero(input == max_value)

    pick_indices = []
    for i in range(max_indices.size(1)):
      pick_indices.append(max_indices[:, i])

    one_hot_encoded = torch.zeros_like(input)
    one_hot_encoded[pick_indices] = 1

    return one_hot_encoded

  else:
    return input

# Checks how many many matrices in two lists of matrices are the same
def count_correct_Higgs(X, Y):

  X_set = {tuple(matrix.flatten().detach().cpu().numpy()) for matrix in X if torch.any(matrix != 0)}
  Y_set = {tuple(matrix.flatten().detach().cpu().numpy()) for matrix in Y}

  count = sum(matrix in Y_set for matrix in X_set)

  return count


# Define PyTorch Dataset
class TestDataset(Dataset):
    def __init__(self, h5_path):
        self.file_path = h5_path
        self.file = h5py.File(self.file_path, "r")

    def __len__(self):
        return len(self.file["INPUTS/Jets/pt"])

    def __getitem__(self, idx):
        # Define a method for extracting a single event input: shape (10, 6) and (10)
        jets_data_pt = torch.tensor(self.file["INPUTS/Jets/pt"][idx])
        jets_data_eta = torch.tensor(self.file["INPUTS/Jets/eta"][idx])
        jets_data_sinphi = torch.tensor(self.file["INPUTS/Jets/sinphi"][idx])
        jets_data_cosphi = torch.tensor(self.file["INPUTS/Jets/cosphi"][idx])
        jets_data_btag = torch.tensor(self.file["INPUTS/Jets/btag"][idx])
        jets_data_invmass = torch.tensor(self.file["INPUTS/Jets/invmass"][idx])
        
        jets_data = torch.stack([jets_data_pt, jets_data_eta, jets_data_sinphi, jets_data_cosphi, jets_data_btag, jets_data_invmass], dim=0)
        jets_data = jets_data.t()
        jets_mask = torch.tensor(self.file["INPUTS/Jets/MASK"][idx], dtype=torch.bool)

        # Define a method for extracting a single event assignments label: shape (3, 10, 10)
        h1_data = self.file['TARGETS']['h1']
        h2_data = self.file['TARGETS']['h2']
        h3_data = self.file['TARGETS']['h3']

        indices_h1 = [self.file['TARGETS']['h1']['b1'][idx], self.file['TARGETS']['h1']['b2'][idx]]
        indices_h2 = [self.file['TARGETS']['h2']['b1'][idx], self.file['TARGETS']['h2']['b2'][idx]]
        indices_h3 = [self.file['TARGETS']['h3']['b1'][idx], self.file['TARGETS']['h3']['b2'][idx]]

        perm_indices_h1 = list(permutations(indices_h1))
        perm_indices_h2 = list(permutations(indices_h2))
        perm_indices_h3 = list(permutations(indices_h3))

        assignments = np.zeros((3, 10, 10), dtype=np.float32)

        if -1 not in indices_h1: assignments[0, perm_indices_h1[0], perm_indices_h1[1]] = 1
        if -1 not in indices_h2: assignments[1, perm_indices_h2[0], perm_indices_h2[1]] = 1
        if -1 not in indices_h3: assignments[2, perm_indices_h3[0], perm_indices_h3[1]] = 1

        assignments = torch.tensor(assignments, dtype=torch.float)

        # Define a method for extracting a single event SvB label: shape (2)
        event_type = self.file['CLASSIFICATIONS']['EVENT']['signal'][idx]
        event_type = torch.tensor(event_type, dtype=torch.int64)

        # Define a method for extracting a single event category: shape (3)
        mask_h1 = h1_data['mask'][idx].astype(np.float32)
        mask_h2 = h2_data['mask'][idx].astype(np.float32)
        mask_h3 = h3_data['mask'][idx].astype(np.float32)

        category = torch.tensor([mask_h1, mask_h2, mask_h3], dtype=torch.float)

        return jets_data, jets_mask, assignments, category, event_type
    

class TestLightning(L.LightningModule):
    def __init__(self, onnx_path):
        super(TestLightning, self).__init__()
        self.session = onnxruntime.InferenceSession(onnx_path)

        self.event_type_counter = torch.zeros(3)  #SETUP MANUALLY
        self.categories_counter = torch.zeros(4)  #SETUP MANUALLY
        self.classification_matrix = torch.zeros(self.event_type_counter.size(0), self.event_type_counter.size(0))
        self.categorization_matrix = torch.zeros(self.categories_counter.size(0), self.categories_counter.size(0))
        self.higgs_purity_matrix = torch.zeros(self.categories_counter.size(0), self.categories_counter.size(0))

    def training_step(self, batch, batch_idx):
       pass

    def validation_step(self, batch, batch_idx):
       pass

    def test_step(self, batch, batch_idx):
        
        jets_data, jets_mask, true_assignments, true_category, true_event_type = batch
        
        input_data = {
           'Jets_data': jets_data.cpu().numpy(),
           'Jets_mask': jets_mask.cpu().numpy()
           }
        output = self.session.run(None, input_data)

        predicted_event_type = np.argmax(output[6], axis=1)
        predicted_event_type = torch.tensor(predicted_event_type)

        predicted_category = np.stack((output[3], output[4], output[5]), axis=1)
        predicted_category = torch.tensor(predicted_category)
        predicted_category = predicted_category.ge(0.5).float()

        predicted_assignments = np.stack((output[0], output[1], output[2]), axis=1)
        predicted_assignments = torch.tensor(predicted_assignments)
        for i in range(predicted_assignments.size(0)):
                for j in range(predicted_assignments.size(1)):
                  predicted_assignments[i][j] = one_hot_encode(predicted_assignments[i][j])
        predicted_assignments = predicted_assignments * predicted_category.unsqueeze(-1).unsqueeze(-1)

        true_category_label = torch.sum(true_category, dim=1).long()
        predicted_category_label = torch.sum(predicted_category, dim=1).long()

        for k in range(true_category_label.size(0)):

            self.event_type_counter[true_event_type[k]] += 1
            self.classification_matrix[true_event_type[k], predicted_event_type[k]] += 1
            
            if true_event_type[k] == 0:
                self.categories_counter[true_category_label[k]] += 1
                self.categorization_matrix[true_category_label[k], predicted_category_label[k]] += 1
            
                self.higgs_purity_matrix[true_category_label[k], predicted_category_label[k]] += count_correct_Higgs(predicted_assignments[k], true_assignments[k])
        
        return
    
    def print_results(self):  # Needs to be setup manually acording to the type of used data and event topology
        type_proportions = torch.zeros_like(self.event_type_counter)
        event_proportions = torch.zeros_like(self.categories_counter)
        type_percentages = torch.zeros_like(self.classification_matrix)
        category_percentages = torch.zeros_like(self.categorization_matrix)
        higgs_percentages = torch.zeros_like(self.higgs_purity_matrix)

        for i in range(type_proportions.size(0)):
          type_proportions[i] = round(self.event_type_counter[i].item() / torch.sum(self.event_type_counter).item(), 4)

        type_proportions = type_proportions.numpy()
        type_proportions_results = PrettyTable()
        type_proportions_results.field_names = ["Process", "HHH", "QCD", "TTToHadronic"]
        type_proportions_results.add_row(["Proportions", type_proportions[0], type_proportions[1], type_proportions[2]])

        print("")
        print("Event type proportions:")
        print(type_proportions_results)

        for i in range(event_proportions.size(0)):
          event_proportions[i] = round(self.categories_counter[i].item() / torch.sum(self.categories_counter).item(), 4)

        event_proportions = event_proportions.numpy()
        event_proportions_results = PrettyTable()
        event_proportions_results.field_names = ["Reconstructible Higgses", "0h", "1h", "2h", "3h"]
        event_proportions_results.add_row(["Proportions", event_proportions[0], event_proportions[1], event_proportions[2], event_proportions[3]])

        print("")
        print("Event proportions (for signal events):")
        print(event_proportions_results)

        for i in range(type_percentages.size(0)):
          for j in range(type_percentages.size(1)):
            type_percentages[i][j] = round(self.classification_matrix[i][j].item() / torch.sum(self.classification_matrix[i]).item(), 4)

        type_percentages = type_percentages.numpy()
        types_results = PrettyTable()
        types_results.field_names = ["True/Predicted", "HHH", "QCD", "TTToHadronic"]

        types_results.add_row(["HHH", type_percentages[0][0], type_percentages[0][1], type_percentages[0][2]])
        types_results.add_row(["QCD", type_percentages[1][0], type_percentages[1][1], type_percentages[1][2]])
        types_results.add_row(["TTToHadronic", type_percentages[1][0], type_percentages[1][1], type_percentages[1][2]])

        print("")
        print("Classification results:")
        print(types_results)

        for i in range(category_percentages.size(0)):
          for j in range(category_percentages.size(1)):
            category_percentages[i][j] = round(self.categorization_matrix[i][j].item() / torch.sum(self.categorization_matrix[i]).item(), 4)
            if self.categorization_matrix[i][j].item() * i != 0:
              higgs_percentages[i][j] = round(self.higgs_purity_matrix[i][j].item() / (torch.sum(self.categorization_matrix[i]).item() * i), 4)
            else:
              higgs_percentages[i][j] = math.nan

        category_percentages = category_percentages.numpy()
        categories_results = PrettyTable()
        categories_results.field_names = ["True/Predicted", "0h", "1h", "2h", "3h"]

        categories_results.add_row(["0h", category_percentages[0][0], category_percentages[0][1], category_percentages[0][2], category_percentages[0][3]])
        categories_results.add_row(["1h", category_percentages[1][0], category_percentages[1][1], category_percentages[1][2], category_percentages[1][3]])
        categories_results.add_row(["2h", category_percentages[2][0], category_percentages[2][1], category_percentages[2][2], category_percentages[2][3]])
        categories_results.add_row(["3h", category_percentages[3][0], category_percentages[3][1], category_percentages[3][2], category_percentages[3][3]])

        print("")
        print("Categorization results (10 Jets):")
        print(categories_results)

        higgs_percentages = higgs_percentages.numpy()
        higgs_results = PrettyTable()
        higgs_results.field_names = ["True/Predicted", "0h", "1h", "2h", "3h"]

        higgs_results.add_row(["0h", higgs_percentages[0][0], higgs_percentages[0][1], higgs_percentages[0][2], higgs_percentages[0][3]])
        higgs_results.add_row(["1h", higgs_percentages[1][0], higgs_percentages[1][1], higgs_percentages[1][2], higgs_percentages[1][3]])
        higgs_results.add_row(["2h", higgs_percentages[2][0], higgs_percentages[2][1], higgs_percentages[2][2], higgs_percentages[2][3]])
        higgs_results.add_row(["3h", higgs_percentages[3][0], higgs_percentages[3][1], higgs_percentages[3][2], higgs_percentages[3][3]])

        print("")
        print("Higgs purity (10 Jets):")
        print(higgs_results)

    def on_test_epoch_end(self):
        self.print_results()

    def configure_optimizers(self):
       pass

def main(onnx_path, h5_path, gpus=None):
    
    # Setup device-agnostic code:
    if gpus is None:
       if torch.cuda.is_available():
          gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device != torch.device("cpu"):
      num_cuda_devices = torch.cuda.device_count()
      if gpus > num_cuda_devices:
        raise ValueError(f'There are only {num_cuda_devices} GPUs available, but requested {gpus}.')
      else:
        gpu_indices = list(range(gpus))
    else:
      gpu_indices = None

    # Initialize dataset
    test_dataset = TestDataset(h5_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=2048, num_workers=28, shuffle=False)
    
    # Initialize Lightning Tester
    tester = L.Trainer(
    max_epochs=1,
    devices=gpu_indices,
    accelerator="auto",
    strategy='ddp_find_unused_parameters_true',
    default_root_dir=None,
    callbacks=False,
    logger=False
)
    
    tapte_lightning = TestLightning(onnx_path)
    print("Lightning Initialized")
    tester.test(tapte_lightning, test_loader)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Does SPANet inference from an ONNX model')
    parser.add_argument('onnx_path', type=str, help='Choose an ONNX model')
    parser.add_argument('-tf', type=str, help='Choose test dataset')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use for testing')

    args = parser.parse_args()

    main(args.onnx_path, args.tf, args.gpus)