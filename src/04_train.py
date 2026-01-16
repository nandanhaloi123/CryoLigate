import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from architecture import SCUNet, CryoEMDataset, CustomLoss

# --- CONFIG ---
HDF5_FILE = "../data/processed/ml_dataset.h5"
CHECKPOINT_DIR = "../checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100

def main():
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"--- Training on {DEVICE} ---")
    dataset = CryoEMDataset(HDF5_FILE)
    train, val = random_split(dataset, [int(0.9*len(dataset)), len(dataset)-int(0.9*len(dataset))])
    
    train_loader = DataLoader(train, batch_size=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val, batch_size=8, shuffle=False)

    model = SCUNet(in_nc=2, ligand_dim=1024).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = CustomLoss()
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for inputs, lig_emb, targets in train_loader:
            inputs, lig_emb, targets = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(inputs, lig_emb)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for inputs, lig_emb, targets in val_loader:
                inputs, lig_emb, targets = inputs.to(DEVICE), lig_emb.to(DEVICE), targets.to(DEVICE)
                preds = model(inputs, lig_emb)
                v_loss += criterion(preds, targets).item()
        
        avg_v = v_loss/len(val_loader)
        print(f"Epoch {epoch+1} | Train: {t_loss/len(train_loader):.4f} | Val: {avg_v:.4f}")
        
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")

if __name__ == "__main__":
    main()
