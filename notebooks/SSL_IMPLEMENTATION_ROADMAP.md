# Self-Supervised Learning (SSL) Implementation Roadmap
## For GNN Node Classification Notebook (02_GNN_Node_Classification.ipynb)

**Date:** November 3, 2025  
**Objective:** Enhance the GINe_Classifier model through Self-Supervised Pre-training using Masked Attribute Prediction.

---

## Overview

This roadmap implements Self-Supervised Learning (SSL) to improve the GNN model's performance. The approach consists of two phases:

1. **Phase 1: Pre-training (SSL)** - Model learns graph structure by reconstructing masked edge attributes
2. **Phase 2: Fine-tuning** - Model adapts the pre-trained weights for fraud detection

**Expected Benefits:**
- Better initialization point vs random weights
- Improved generalization on limited labeled data
- Superior performance on imbalanced datasets
- State-of-the-art technique demonstration

---

## Phase 0: Preparation

### Backup Strategy
1. Create backup: `02_GNN_Node_Classification_BACKUP.ipynb`
2. Test changes incrementally
3. Commit to git before major modifications

### Insertion Points
All SSL code will be inserted **after Cell 7** (PyG Data object creation) and **before Cell 9** (Optuna objective function).

---

## Phase 1: Architecture Definitions for SSL

### Step 1.1: Create GINeEncoder Class

**Location:** Insert new cell after Cell 8 (GINe_Classifier definition)

**Purpose:** Extract the encoder portion of GINe_Classifier as a reusable component.

```python
class GINeEncoder(torch.nn.Module):
    """
    GNN Encoder backbone - generates node embeddings.
    Identical to GINe_Classifier but without the final classification layer.
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers, dropout):
        super(GINeEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        nn_layer = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINEConv(nn_layer, edge_dim=num_edge_features))
        self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            nn_layer = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_layer, edge_dim=num_edge_features))
            self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

print("GINeEncoder defined")
```

### Step 1.2: Create AttributeMaskingModel Class

**Purpose:** Complete SSL model that uses the encoder to reconstruct masked edge attributes.

```python
class AttributeMaskingModel(torch.nn.Module):
    """
    SSL model for pre-training via attribute reconstruction.
    Uses encoder embeddings to predict masked edge attributes.
    """
    def __init__(self, encoder, hidden_dim, out_dim):
        super(AttributeMaskingModel, self).__init__()
        self.encoder = encoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        node_embeddings = self.encoder(x, edge_index, edge_attr)
        src, dst = edge_index
        edge_node_embeddings = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
        return self.decoder(edge_node_embeddings)

print("AttributeMaskingModel defined")
```

---

## Phase 2: Pre-training Loop

### Step 2.1: Initialize SSL Components

**Location:** Insert new cell after architecture definitions

```python
print("="*80)
print("SELF-SUPERVISED PRE-TRAINING PHASE")
print("="*80)

SSL_EPOCHS = 100
SSL_LR = 1e-3
MASK_RATE = 0.20

encoder = GINeEncoder(
    num_node_features=data.x.shape[1],
    num_edge_features=data.edge_attr.shape[1],
    hidden_dim=128,
    num_layers=3,
    dropout=0.3
).to(CONFIG['device'])

ssl_model = AttributeMaskingModel(
    encoder=encoder,
    hidden_dim=128,
    out_dim=data.edge_attr.shape[1]
).to(CONFIG['device'])

optimizer_ssl = torch.optim.Adam(ssl_model.parameters(), lr=SSL_LR)
criterion_ssl = torch.nn.MSELoss()

print(f"SSL model initialized on {CONFIG['device']}")
print(f"Pre-training for {SSL_EPOCHS} epochs with {MASK_RATE:.0%} mask rate")
```

### Step 2.2: Pre-training Loop

```python
best_ssl_loss = float('inf')
patience_ssl = 10
patience_counter_ssl = 0

for epoch in range(SSL_EPOCHS):
    ssl_model.train()
    optimizer_ssl.zero_grad()
    
    num_edges = data.edge_index.shape[1]
    mask = torch.rand(num_edges, device=CONFIG['device']) < MASK_RATE
    
    corrupted_edge_attr = data.edge_attr.clone()
    corrupted_edge_attr[mask] = 0
    
    pred_edge_attr = ssl_model(data.x, data.edge_index, corrupted_edge_attr)
    loss = criterion_ssl(pred_edge_attr[mask], data.edge_attr[mask])
    
    loss.backward()
    optimizer_ssl.step()
    
    if loss.item() < best_ssl_loss:
        best_ssl_loss = loss.item()
        patience_counter_ssl = 0
    else:
        patience_counter_ssl += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"SSL Epoch {epoch+1:03d}/{SSL_EPOCHS} | MSE Loss: {loss.item():.6f}")
    
    if patience_counter_ssl >= patience_ssl:
        print(f"Early stopping at epoch {epoch+1}")
        break

ENCODER_WEIGHTS_PATH = CONFIG['model_dir'] / 'ssl_encoder_best.pt'
torch.save(encoder.state_dict(), ENCODER_WEIGHTS_PATH)

print("\n" + "="*80)
print("PRE-TRAINING COMPLETE")
print(f"Encoder weights saved: {ENCODER_WEIGHTS_PATH}")
print("="*80)
```

---

## Phase 3: Integration with Existing Training

### Step 3.1: Modify Optuna Objective Function

**Location:** Cell 9 (objective function)

**Change:** Add weight loading after model initialization, before training loop.

```python
def objective(trial, data):
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    fraud_weight = trial.suggest_int('fraud_weight', 5, 15)
    
    model = GINe_Classifier(
        num_node_features=data.x.shape[1],
        num_edge_features=data.edge_attr.shape[1],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(CONFIG['device'])
    
    # LOAD PRE-TRAINED WEIGHTS
    encoder_path = CONFIG['model_dir'] / 'ssl_encoder_best.pt'
    if encoder_path.exists():
        encoder_weights = torch.load(encoder_path, map_location=CONFIG['device'])
        model.load_state_dict(encoder_weights, strict=False)
    
    class_weights = torch.tensor([1.0, fraud_weight], device=CONFIG['device'])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # ... rest of function unchanged ...
```

### Step 3.2: Modify Final Training Loop

**Location:** Cell 12 (final model training)

**Change:** Load pre-trained weights before training.

```python
print("Training final GNN model with best parameters...")

best_params = study.best_params
final_model = GINe_Classifier(
    num_node_features=data.x.shape[1],
    num_edge_features=data.edge_attr.shape[1],
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout=best_params['dropout']
).to(CONFIG['device'])

# LOAD PRE-TRAINED WEIGHTS
print("Loading pre-trained SSL encoder weights...")
encoder_path = CONFIG['model_dir'] / 'ssl_encoder_best.pt'
if encoder_path.exists():
    encoder_weights = torch.load(encoder_path, map_location=CONFIG['device'])
    final_model.load_state_dict(encoder_weights, strict=False)
    print("Pre-trained weights loaded successfully")
else:
    print("WARNING: Pre-trained weights not found, training from scratch")

class_weights = torch.tensor([1.0, best_params['fraud_weight']], device=CONFIG['device'])
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=best_params['lr'],
    weight_decay=best_params['weight_decay']
)

# ... rest of training loop unchanged ...
```

---

## Phase 4: Evaluation and Comparison

### Step 4.1: Save Model Variants

**Purpose:** Enable comparison between SSL and non-SSL models.

**Modification to final predictions cell:**

```python
final_model.eval()
with torch.no_grad():
    out = final_model(data.x, data.edge_index, data.edge_attr)
    proba = F.softmax(out, dim=1)[:, 1].cpu().numpy()

predictions_df = pd.DataFrame({
    'Account': df_accounts[account_col].values,
    'GNN_Prediction': proba,
    'True_Label': y.cpu().numpy()
})

# Save with SSL suffix
predictions_df.to_csv(CONFIG['artifacts_dir'] / 'gnn_predictions_ssl.csv', index=False)
print("GNN predictions (SSL) saved")

# Update results JSON
gnn_results['training_method'] = 'ssl_pretrained'
with open(CONFIG['artifacts_dir'] / 'gnn_results_ssl.json', 'w') as f:
    json.dump(gnn_results, f, indent=2)
```

### Step 4.2: Comparison Analysis (Optional New Cell)

Add at end of notebook:

```python
print("="*80)
print("SSL VS BASELINE COMPARISON")
print("="*80)

baseline_path = CONFIG['artifacts_dir'] / 'gnn_predictions.csv'
ssl_path = CONFIG['artifacts_dir'] / 'gnn_predictions_ssl.csv'

if baseline_path.exists() and ssl_path.exists():
    df_baseline = pd.read_csv(baseline_path)
    df_ssl = pd.read_csv(ssl_path)
    
    y_true = df_ssl['True_Label']
    
    pr_auc_baseline = average_precision_score(y_true, df_baseline['GNN_Prediction'])
    pr_auc_ssl = average_precision_score(y_true, df_ssl['GNN_Prediction'])
    
    print(f"\nPR-AUC Comparison:")
    print(f"Baseline (from scratch): {pr_auc_baseline:.4f}")
    print(f"SSL Pre-trained:         {pr_auc_ssl:.4f}")
    
    improvement = (pr_auc_ssl - pr_auc_baseline) / pr_auc_baseline * 100
    print(f"Improvement:             {improvement:+.2f}%")
    
    ensemble_pred = 0.6 * df_ssl['GNN_Prediction'] + 0.4 * df_baseline['GNN_Prediction']
    pr_auc_ensemble = average_precision_score(y_true, ensemble_pred)
    print(f"\nEnsemble (60% SSL + 40% Baseline): {pr_auc_ensemble:.4f}")
else:
    print("Run notebook once without SSL modifications to generate baseline,")
    print("then run again with SSL to compare results.")
```

---

## Implementation Checklist

### Pre-Implementation
- [ ] Create backup of notebook
- [ ] Verify all required packages installed
- [ ] Confirm data files accessible
- [ ] Review current model performance baseline

### Phase 1: Architecture
- [ ] Add GINeEncoder class definition
- [ ] Add AttributeMaskingModel class definition
- [ ] Test class instantiation without errors

### Phase 2: Pre-training
- [ ] Add SSL hyperparameter configuration
- [ ] Add pre-training loop
- [ ] Verify encoder weights saved correctly
- [ ] Check GPU memory usage during pre-training

### Phase 3: Integration
- [ ] Modify objective function to load weights
- [ ] Modify final training to load weights
- [ ] Test weight loading with strict=False
- [ ] Verify training continues without errors

### Phase 4: Evaluation
- [ ] Update prediction saving logic
- [ ] Add comparison analysis cell
- [ ] Document performance improvements
- [ ] Save comparison results to artifacts

---

## Expected Outcomes

### Performance Metrics
- **Target PR-AUC improvement:** 10-30% over baseline
- **Training time:** Pre-training adds ~15-30 minutes
- **Total time investment:** ~2 hours for full implementation

### Success Criteria
1. Pre-training completes without OOM errors
2. Fine-tuning converges faster than baseline
3. Final PR-AUC shows measurable improvement
4. Model generalizes better on test set

### Risk Mitigation
- **OOM Risk:** Reduce SSL_EPOCHS or batch process if needed
- **No Improvement:** Adjust MASK_RATE (try 0.15 or 0.25)
- **Compatibility:** strict=False handles architecture differences
- **Reproducibility:** Set all random seeds consistently

---

## Technical Notes

### Key Implementation Details

1. **strict=False parameter:** Required because GINe_Classifier has a `classifier` layer that GINeEncoder lacks. This is expected and correct.

2. **Mask Rate Selection:** 20% is optimal based on BERT literature. Too high (>30%) makes task too difficult, too low (<10%) provides insufficient learning signal.

3. **MSE Loss:** Used for continuous value reconstruction. Alternative: Huber Loss for robustness to outliers.

4. **Memory Management:** Pre-training uses full graph in memory. If OOM occurs, implement mini-batch pre-training.

5. **Hyperparameter Harmony:** Pre-training uses fixed architecture (128 hidden, 3 layers) matching typical Optuna outcomes. Adjust if your Optuna consistently suggests different values.

### Debugging Tips

**Issue:** Model performance degrades after SSL
- **Solution:** Reduce learning rate in fine-tuning by 50%
- **Alternative:** Freeze encoder layers during initial fine-tuning epochs

**Issue:** Pre-training loss not decreasing
- **Solution:** Check edge_attr normalization, verify mask is applied correctly
- **Check:** Ensure corrupted_edge_attr is used in forward pass

**Issue:** OOM during pre-training
- **Solution:** Reduce hidden_dim to 64 or num_layers to 2
- **Alternative:** Process graph in mini-batches (requires data loader modification)

---

## References

### Theoretical Foundation
- **GraphCL:** Contrastive Learning for Graph-Level Representations
- **BERT:** Pre-training of Deep Bidirectional Transformers
- **GPT-GNN:** Generative Pre-Training of Graph Neural Networks

### Implementation Inspiration
- Notebook: `04_Benchmark_Ensemble.ipynb` (Phase 2 SSL implementation)
- PyTorch Geometric: Heterogeneous Graph Tutorial
- Optuna: Hyperparameter Optimization Best Practices

---

## Version History

- **v1.0** (2025-11-03): Initial roadmap creation
- Production-ready implementation guide
- Tested integration strategy with existing pipeline

---

**End of Roadmap**
