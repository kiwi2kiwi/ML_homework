import torch
import pickle
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from torch_geometric.data import DataLoader # version was 2.0.2

import numpy as np
import matplotlib.pyplot as plt
import gnn_arch_ES

test_loader = torch.load("../data/test_data.ldr")
model = torch.load("../GNN_models/ES_best.model")
device = "cpu"
# Assuming 'model' is your trained GNN model and 'test_loader' is your test data loader
test_batch = next(iter(test_loader))
with torch.no_grad():
  test_batch.to(device)
  pred, embed = model.forward(test_batch.x.float(), test_batch.edge_index, test_batch.batch)

  # Get node-level predictions
  node_predictions = pred

  # Assuming your model predicts a single property per node
  predicted_properties = node_predictions.cpu().numpy().flatten()

  # Get the SMILES string for the molecule (assuming it's available in your data)
  smiles = test_batch.smiles[0]

  # Create an RDKit molecule object
  molecule = Chem.MolFromSmiles(smiles)
  print(len(predicted_properties))

  predicted_properties = predicted_properties[:molecule.GetNumAtoms()]
  actual_properties = test_batch.y.float().cpu().numpy().flatten()[:molecule.GetNumAtoms()]
  print(molecule.GetNumAtoms())
  print(len(predicted_properties))

  # Check if the number of atoms matches the number of predictions
  if molecule.GetNumAtoms() != len(predicted_properties):
    print(f"Warning: Mismatch in number of atoms ({molecule.GetNumAtoms()}) and predictions ({len(predicted_properties)})")
    # Handle the mismatch, e.g., truncate predictions or skip the molecule
  else:
    # Add predicted properties to the molecule as atom properties
    for i, prop in enumerate(predicted_properties):
      var1 = molecule.GetAtomWithIdx(i)
      var1.SetProp('atomNote', str(round(prop, 3)))

  # Display the molecule with predicted properties
  IPythonConsole.drawOptions.addAtomIndices = True
  display(molecule)

############## ERROR ON THE ELEMENTS

test_batch = next(iter(test_loader))
test_batch.x.shape[0]/test_batch.y.shape[0]
test_batch.x # node level features
test_batch.y # graph level target
molecules = test_batch.smiles
molec = molecules[0]
len(molecules)
len(molec)
flat_molecules = [item for sublist in molecules for item in sublist]
from collections import Counter
element_counts = Counter(flat_molecules)
print(element_counts)
atom_list = [atom.GetSymbol() for atom in Chem.MolFromSmiles(molec).GetAtoms()]
print(atom_list)



with torch.no_grad():
    test_batch.to(device)
    pred, embed = model.forward(test_batch.x.float(), test_batch.edge_index, test_batch.batch)

    # Get node-level predictions
    node_predictions = pred

    # Assuming your model predicts a single property per node
    predicted_properties = node_predictions.cpu().numpy().flatten()

# Assuming 'actual_properties' is a tensor of graph-level properties and 'test_batch' is your batch data
actual_properties = test_batch.y.float()#.cpu().numpy().flatten()

# Repeat the graph-level properties for each node in the graph
num_nodes_in_batch = test_batch.batch.shape[0]  # Total number of nodes
num_graphs_in_batch = actual_properties.shape[0]  # Number of graphs (batch size)

# Create a mapping from node index to graph index
node_to_graph_map = torch.zeros(num_nodes_in_batch, dtype=torch.int64)

# Iterate and assign graph indices to nodes
for graph_index in range(num_graphs_in_batch):
  node_to_graph_map[test_batch.batch == graph_index] = graph_index

# Repeat the graph-level properties for each node based on the mapping
repeated_actual_properties = actual_properties[node_to_graph_map]

errors = predicted_properties - repeated_actual_properties.cpu().numpy().flatten()


mol = Chem.MolFromSmiles(molec)
atoms = {}
ctr = 0
for molecule in molecules:
  for atom in Chem.MolFromSmiles(molecule).GetAtoms():
      # print(atom.GetIdx(), ": ", atom.GetSymbol(), " error: ", )
      if atom.GetSymbol() not in atoms.keys():
        atoms[atom.GetSymbol()] = [1,[np.abs(errors[ctr])]]
      else:
        atoms[atom.GetSymbol()][0] += 1
        atoms[atom.GetSymbol()][1].append(np.abs(errors[ctr]))# += np.abs(errors[ctr])
      ctr+=1

[print(key, " average error: ", "%.3f" % (np.mean(atoms[key][1])), " std: ", "%.3f" % np.std(atoms[key][1])) for key in atoms.keys()]

# TODO  get error for edges


#[print(key, " average error: ", "%.3f" % (atoms[key][0]/atoms[key][1])) for key in atoms.keys()]
# print("Atom counts: ", atoms)
# atom_cts = Counter(atoms)
#print(atom_cts)
#    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))

# print([atom.GetSymbol() for atom in Chem.MolFromSmiles(molec).GetAtoms()])

import rdkit
#rdkit.Chem.Draw.DebugDraw(molecule, size=(350, 350), drawer=None, asSVG=True, useBW=True, includeHLabels=True, addAtomIndices=True, addBondIndices=False)



############ ERROR ON THE EDGES

import numpy as np

with torch.no_grad():
    test_batch.to(device)
    pred, embed = model.forward(test_batch.x.float(), test_batch.edge_index, test_batch.batch)

    # Get node-level predictions
    node_predictions = pred

    # Assuming your model predicts a single property per node
    predicted_properties = node_predictions.cpu().numpy().flatten()

# Assuming 'actual_properties' is a tensor of graph-level properties and 'test_batch' is your batch data
actual_properties = test_batch.y.float()  # .cpu().numpy().flatten()

# Repeat the graph-level properties for each node in the graph
num_nodes_in_batch = test_batch.batch.shape[0]  # Total number of nodes
num_graphs_in_batch = actual_properties.shape[0]  # Number of graphs (batch size)

# Create a mapping from node index to graph index
node_to_graph_map = torch.zeros(num_nodes_in_batch, dtype=torch.int64)

# Iterate and assign graph indices to nodes
for graph_index in range(num_graphs_in_batch):
    node_to_graph_map[test_batch.batch == graph_index] = graph_index

# Repeat the graph-level properties for each node based on the mapping
repeated_actual_properties = actual_properties[node_to_graph_map]

errors = predicted_properties - repeated_actual_properties.cpu().numpy().flatten()

# Get the edge indices
edge_index = test_batch.edge_index.cpu().numpy()


# eliminate edges that are the same, but viewed from the other side
def eliminate_flipped_edges(edge_index):
    # Convert the edge_index tensor to a list of tuples
    edges = [(edge_index[0, i].item(), edge_index[1, i].item()) for i in range(edge_index.shape[1])]

    # Create a set to store the unique edges
    unique_edges = set()

    for edge in edges:
        if (edge[1], edge[0]) not in unique_edges and (edge[0], edge[1]) not in unique_edges:
            unique_edges.add(edge)

    # Iterate over the edges and add only the unique ones to the set
    # for edge in edges:
    #   if (edge[1], edge[0]) not in unique_edges:
    #     unique_edges.add(edge)
    cleaned_edges = []
    for edge in edges:
        if edge in unique_edges:
            cleaned_edges.append(edge)

    unique_edge_index = torch.as_tensor(cleaned_edges).T#torch.tensor(cleaned_edges).T

    # Convert the set of unique edges back to a tensor
    # unique_edge_index = torch.tensor([[edge[0] for edge in unique_edges],
    #                                   [edge[1] for edge in unique_edges]])
    return unique_edge_index


# Example usage with the test_batch
unique_edge_index = eliminate_flipped_edges(edge_index)
print(unique_edge_index)

# Create a dictionary to store edge errors
edge_errors = {}

# Calculate the error for each edge
for i in range(unique_edge_index.shape[1]):
    source_node = unique_edge_index[0, i]
    target_node = unique_edge_index[1, i]
    edge_error = np.abs(errors[source_node]) + np.abs(errors[target_node])/2

    edge_errors[(source_node, target_node)] = edge_error

print(source_node)
print(target_node)
print("sz:", predicted_properties.size)
len(edge_errors.keys())

molecules = test_batch.smiles
atom_list = []
for molecule in molecules:
    [atom_list.append(at.GetSymbol()) for at in Chem.MolFromSmiles(molecule).GetAtoms()]

edge_type = {}
# TODO take the edge
for i in edge_errors.keys():
    i[1]
    atom_list[i[1]]
    if (atom_list[i[0]], atom_list[i[1]]) in edge_type.keys():
        edge_type[(atom_list[i[0]], atom_list[i[1]])].append(edge_errors[i])
    else:
        edge_type[(atom_list[i[0]], atom_list[i[1]])] = [edge_errors[i]]

cleaned_edge_type={}
for i in edge_type.keys():
    if (i[1],i[0]) not in cleaned_edge_type.keys():
        cleaned_edge_type[i] = edge_type[i]
    else:
        cleaned_edge_type[(i[1],i[0])].extend(edge_type[i])

[print(key, " average error: ", "%.3f" % (np.mean(cleaned_edge_type[key])), " std: ", "%.3f" % np.std(cleaned_edge_type[key]), " len: ", len(cleaned_edge_type[key])) for key
 in cleaned_edge_type.keys()]
sum([len(cleaned_edge_type[key]) for key in cleaned_edge_type.keys()])
# how many C S edges? have we lost some?
["P" if "P" in [atom.GetSymbol() for atom in Chem.MolFromSmiles(molecule).GetAtoms()] else "" for molecule in molecules]
from rdkit.Chem import Draw
import io

for molecule in molecules:
    if "P" in [atom.GetSymbol() for atom in Chem.MolFromSmiles(molecule).GetAtoms()]:
        print("found P")
        print(molecule)

        # Assume `molecule` is your RDKit molecule object
        img = Draw.MolToImage(Chem.MolFromSmiles(molecule))

        # Convert the image to a format that matplotlib can handle (e.g., PNG)
        img_buf = io.BytesIO()
        img.save(img_buf, format='PNG')
        img_buf.seek(0)

        # Display the image using matplotlib
        img = plt.imread(img_buf)
        plt.imshow(img)
        plt.axis('off')  # Turn off the axis
        plt.show()
plt.show()
print("test")