import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score
import json

class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    # Simplified loader without neighbor sampling to avoid dependency issues
    from torch.utils.data import DataLoader, TensorDataset

    if isinstance(tr_data, HeteroData):
        # For heterogeneous data, create simple tensor datasets
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y

        # Create simple datasets
        tr_dataset = TensorDataset(tr_inds, tr_edge_label[tr_inds])
        val_dataset = TensorDataset(val_inds, val_data['node', 'to', 'node'].y[val_inds])
        te_dataset = TensorDataset(te_inds, te_data['node', 'to', 'node'].y[te_inds])

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        # For homogeneous data, create simple tensor datasets
        tr_dataset = TensorDataset(tr_inds, tr_data.y[tr_inds])
        val_dataset = TensorDataset(val_inds, val_data.y[val_inds])
        te_dataset = TensorDataset(te_inds, te_data.y[te_inds])

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False)

    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args):
    '''Simplified evaluation for homogeneous graph data.'''
    model.eval()

    # Move data to device
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
    if edge_attr is not None and edge_attr.shape[1] > 1:
        edge_attr = edge_attr[:, 1:]  # Remove edge ID

    # Forward pass
    out = model(x, edge_index, edge_attr)
    pred = out[inds]
    ground_truth = data.y[inds].to(device)

    # Calculate predictions
    pred_classes = pred.argmax(dim=-1).cpu().numpy()
    gt_classes = ground_truth.cpu().numpy()
    f1 = f1_score(gt_classes, pred_classes)

    # Save predictions if requested
    import pandas as pd
    import os

    if hasattr(args, 'save_model') and args.save_model:
        print("💾 Salvando predições...")

        output_df = pd.DataFrame({
            'prediction_prob': pred[:, 1].cpu().numpy() if pred.shape[1] > 1 else pred[:, 0].cpu().numpy(),
            'ground_truth': gt_classes
        })

        output_path = os.path.join(os.getcwd(), "multi_gnn_predictions.csv")
        output_df.to_csv(output_path, index=False)
        print(f"✅ Predições salvas em: {output_path}")

    return f1

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args):
    '''Evaluates the model performane for heterogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch['node'].n_id
            add_edge_index = data['node', 'to', 'node'].edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data['node', 'to', 'node'].edge_attr[missing_ids, :].detach().clone()
            add_y = data['node', 'to', 'node'].y[missing_ids].detach().clone()
        
            batch['node', 'to', 'node'].edge_index = torch.cat((batch['node', 'to', 'node'].edge_index, add_edge_index), 1)
            batch['node', 'to', 'node'].edge_attr = torch.cat((batch['node', 'to', 'node'].edge_attr, add_edge_attr), 0)
            batch['node', 'to', 'node'].y = torch.cat((batch['node', 'to', 'node'].y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    # --- INÍCIO DA MODIFICAÇÃO PARA SALVAR PREDIÇÕES ---
    import pandas as pd
    import os

    # Salvar apenas durante a avaliação final (não na validação)
    # Usamos uma heurística baseada no tamanho do batch
    is_test_evaluation = len(preds) > 1000  # Heurística simples

    if is_test_evaluation and hasattr(args, 'save_model') and args.save_model:
        print("💾 Salvando predições do conjunto de teste...")

        # Combinar todas as predições e ground truths
        all_preds = torch.cat(preds, dim=0).cpu().numpy()
        all_ground_truths = torch.cat(ground_truths, dim=0).cpu().numpy()

        # Criar DataFrame com predições
        output_df = pd.DataFrame({
            'prediction_prob': all_preds[:, 1] if len(all_preds.shape) > 1 else all_preds,  # Probabilidade da classe 1
            'ground_truth': all_ground_truths
        })

        output_path = os.path.join(os.getcwd(), "multi_gnn_predictions.csv")
        output_df.to_csv(output_path, index=False)
        print(f"✅ Predições salvas em: {output_path}")
    # --- FIM DA MODIFICAÇÃO ---

    return f1

def save_model(model, optimizer, epoch, args, data_config):
    # Generate unique name if not provided
    model_name = args.unique_name if args.unique_name else f"{args.model}_{args.data}_{epoch}"
    
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'{data_config["model_to_save"]}/checkpoint_{model_name}{"" if not args.finetune else "_finetuned"}.tar')
    
def load_model(model, device, args, config, data_config):
    checkpoint = torch.load(f'{data_config["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer