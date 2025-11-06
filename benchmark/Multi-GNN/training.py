import torch
import tqdm
from sklearn.metrics import f1_score
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model
from models import GINe, PNA, GATe, RGCN
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero, summary
from torch_geometric.utils import degree
import wandb
import logging

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    # Simplified training without complex batching
    best_val_f1 = 0

    for epoch in range(config.epochs):
        # Training step - use full graph
        model.train()
        optimizer.zero_grad()

        # Move data to device
        x = val_data.x.to(device)
        edge_index = val_data.edge_index.to(device)
        edge_attr = val_data.edge_attr.to(device) if hasattr(val_data, 'edge_attr') else None
        if edge_attr is not None and edge_attr.shape[1] > 1:
            edge_attr = edge_attr[:, 1:]  # Remove edge ID

        # Forward pass on training edges only
        out = model(x, edge_index, edge_attr)
        pred = out[tr_inds]
        ground_truth = val_data.y[tr_inds].to(device)

        loss = loss_fn(pred, ground_truth)
        loss.backward()
        optimizer.step()

        # Calculate training F1
        pred_classes = pred.argmax(dim=-1).cpu().numpy()
        gt_classes = ground_truth.cpu().numpy()
        train_f1 = f1_score(gt_classes, pred_classes)

        logging.info(f'Epoch {epoch}: Train Loss: {loss:.4f}, Train F1: {train_f1:.4f}')

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index, edge_attr)
            val_pred = val_out[val_inds]
            val_gt = val_data.y[val_inds].to(device)
            val_pred_classes = val_pred.argmax(dim=-1).cpu().numpy()
            val_gt_classes = val_gt.cpu().numpy()
            val_f1 = f1_score(val_gt_classes, val_pred_classes)

            # Test - usar te_data em vez de val_data
            te_x = te_data.x.to(device)
            te_edge_index = te_data.edge_index.to(device)
            te_edge_attr = te_data.edge_attr.to(device) if hasattr(te_data, 'edge_attr') else None
            if te_edge_attr is not None and te_edge_attr.shape[1] > 1:
                te_edge_attr = te_edge_attr[:, 1:]  # Remove edge ID

            te_out = model(te_x, te_edge_index, te_edge_attr)
            te_pred = te_out[te_inds]
            te_gt = te_data.y[te_inds].to(device)
            te_pred_classes = te_pred.argmax(dim=-1).cpu().numpy()
            te_gt_classes = te_gt.cpu().numpy()
            te_f1 = f1_score(te_gt_classes, te_pred_classes)

        logging.info(f'Validation F1: {val_f1:.4f}, Test F1: {te_f1:.4f}')

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)

    # Save final model
    if args.save_model:
        save_model(model, optimizer, config.epochs - 1, args, data_config)

    return model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    #training
    best_val_f1 = 0
    for epoch in range(config.epochs):
        total_loss = total_examples = 0
        preds = []
        ground_truths = []
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            #select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            #remove the unique edge id from the edge features, as it's no longer needed
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            pred = out[mask]
            ground_truth = batch['node', 'to', 'node'].y[mask]
            preds.append(pred.argmax(dim=-1))
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            loss = loss_fn(pred, ground_truth)

            loss.backward()
            optimizer.step()

            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
            
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)
        logging.info(f'Train F1: {f1:.4f}')

        #evaluate
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)
        wandb.log({"f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    # Save model at the end of training if requested
    if args.save_model:
        save_model(model, optimizer, config.epochs - 1, args, data_config)
    
    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    if args.model == "gin":
        model = GINe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "gat":
        model = GATe(
                num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
                n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
                edge_updates=args.emlps, edge_dim=e_dim,
                dropout=config.dropout, final_dropout=config.final_dropout
                )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
            )
    elif config.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None #(maybe)
        )
    
    return model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #define a model config dictionary and wandb logging at the same time
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name", #replace this with your wandb project name if you want to use wandb logging

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    # Skip complex loaders and use simplified approach
    # tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)
    tr_loader = val_loader = te_loader = None  # Not used in simplified training

    #get the model - use full graph data instead of batch
    # Create a dummy batch with full graph structure
    class DummyBatch:
        def __init__(self, data):
            self.x = data.x
            self.edge_index = data.edge_index
            self.edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            self.y = data.y if hasattr(data, 'y') else None
        def to(self, device):
            self.x = self.x.to(device)
            self.edge_index = self.edge_index.to(device)
            if self.edge_attr is not None:
                self.edge_attr = self.edge_attr.to(device)
            if self.y is not None:
                self.y = self.y.to(device)
            return self

    sample_batch = DummyBatch(tr_data)
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    sample_batch.to(device)
    sample_x = sample_batch.x
    sample_edge_index = sample_batch.edge_index
    sample_edge_attr = sample_batch.edge_attr
    if sample_edge_attr is not None and sample_edge_attr.shape[1] > 1:
        sample_edge_attr = sample_edge_attr[:, 1:]  # Remove edge ID if present
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))

    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()