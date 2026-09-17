import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

class SimCLRTransform:
    def __init__(self, size=224):
        # Strong augmentations for SimCLR
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            # Fallback for corrupted images
            img = Image.new('RGB', (224, 224))
        if self.transform:
            return self.transform(img)
        return img

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def nt_xent_loss(z1, z2, temperature=0.5):
    # z1, z2: [B, D]
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0) # [2B, D]
    z = F.normalize(z, dim=1)
    
    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature # [2B, 2B]
    
    # Remove self-similarity
    sim_matrix.fill_diagonal_(-1e9)
    
    # Targets: z1[i] should match z2[i], which is at index i+B
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)], dim=0).to(z.device)
    
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Gather all images
    base_dir = '/Volumes/Macintosh SUB/Dataset/yolo_data'
    image_paths = []
    for split in ['train', 'val', 'test']:
        paths = glob.glob(os.path.join(base_dir, split, 'images', '*.*'))
        image_paths.extend(paths)
    print(f"Total images for SSL: {len(image_paths)}")

    dataset = UnlabeledImageDataset(image_paths, transform=SimCLRTransform(size=224))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)

    # Load YOLO model (from scratch YOLO11s)
    yolo = YOLO('yolo11s.pt')
    yolo_model = yolo.model.to(device)
    
    # We will freeze the head and only train the backbone
    # However, since gradients only flow from the hook, head weights won't update anyway.
    
    # Setup Hook to capture SPPF output (Layer 9 in YOLOv8n)
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output
        return hook
        
    hook_handle = yolo_model.model[9].register_forward_hook(get_features('sppf'))

    # The output channels of SPPF in YOLOv8n is usually 256. 
    # Let's dynamically check by passing a dummy tensor.
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    _ = yolo_model(dummy_input)
    dummy_feat = features['sppf']
    in_channels = dummy_feat.size(1)
    print(f"Backbone output channels: {in_channels}")

    proj_head = ProjectionHead(in_dim=in_channels, out_dim=128).to(device)

    # Optimizer
    # We optimize both the YOLO model parameters (backbone) and the projection head
    optimizer = optim.Adam([
        {'params': yolo_model.model[:10].parameters()},
        {'params': proj_head.parameters()}
    ], lr=1e-3)

    epochs = 50
    print("Starting SimCLR training from scratch...")
    
    for epoch in range(epochs):
        yolo_model.train()
        proj_head.train()
        total_loss = 0
        
        for i, (view1, view2) in enumerate(dataloader):
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            # Forward view 1
            _ = yolo_model(view1)
            feat1 = features['sppf'] # [B, C, H, W]
            feat1 = feat1.mean(dim=[2, 3]) # Global Average Pooling -> [B, C]
            z1 = proj_head(feat1)
            
            # Forward view 2
            _ = yolo_model(view2)
            feat2 = features['sppf']
            feat2 = feat2.mean(dim=[2, 3])
            z2 = proj_head(feat2)
            
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f} ---")
        
    print("SimCLR training complete.")
    
    # Save the updated YOLO model
    print("Saving pretrained model...")
    hook_handle.remove() # Remove the hook so the model can be pickled
    yolo.ckpt['model'] = yolo_model.half() # Save model in fp16 as Ultralytics does
    torch.save(yolo.ckpt, 'models/simclr_scratch_yolo11s.pt')
    print("Saved pretrained model to models/simclr_scratch_yolo11s.pt")

if __name__ == '__main__':
    main()
