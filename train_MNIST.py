import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphNorm
from torch_geometric.utils import from_networkx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import networkx as nx
import os
import warnings
import torchvision
import torchvision.transforms as transforms
warnings.filterwarnings("ignore")

SEED = 2025
IMG_SIZE = 28
NUM_NODES = IMG_SIZE * IMG_SIZE

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# 创建保存目录
os.makedirs("results_mnist_revisedR2/best_cases", exist_ok=True)
os.makedirs("data_cache_mnist", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "figure.dpi": 300,
    "axes.grid": False,
})


class MNISTGraphLoader:
    def __init__(self):
        self.input_snr_val = 0.0

    def load(self):
        print("\n>>> [Step 1] Loading MNIST Data...")

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.MNIST(root='./data_cache_mnist', train=True,
                                              download=True, transform=transform)

        images = trainset.data[:1000].float() / 255.0
        X_clean = images.view(-1, NUM_NODES).unsqueeze(-1).to(device)
        print(f"   √ Loaded {len(X_clean)} images.")

        print("   Constructing 28x28 Grid Graph...")
        G = nx.grid_2d_graph(IMG_SIZE, IMG_SIZE)
        G = nx.convert_node_labels_to_integers(G)
        edge_index = from_networkx(G).edge_index.to(device)

        print("   Generating Block Masking Task (Center Hole)...")
        X_noisy = X_clean.clone()

        mask_size = 12
        start = (IMG_SIZE - mask_size) // 2
        end = start + mask_size

        mask_matrix = torch.ones(IMG_SIZE, IMG_SIZE).to(device)
        mask_matrix[start:end, start:end] = 0
        mask_flat = mask_matrix.view(-1, 1)

        X_noisy = X_clean * mask_flat

        noise_power = torch.sum((X_noisy - X_clean) ** 2)
        sig_power = torch.sum(X_clean ** 2)
        self.input_snr_val = 10 * torch.log10(sig_power / (noise_power + 1e-6)).item()

        print(f"\n{'=' * 40}")
        print(f"DATASET: MNIST (Real Images)")
        print(f"Input SNR: {self.input_snr_val:.2f} dB")
        print(f"{'=' * 40}\n")

        return X_noisy, X_clean, edge_index, NUM_NODES


class SpectralSerializer:
    def __init__(self):
        self.perm = None
        self.inv_perm = None
        self.pe = None

        self.rand_perm = None
        self.inv_rand_perm = None

        self.dw_perm = None
        self.inv_dw_perm = None

    def compute_ordering(self, edge_index, num_nodes):
        print("   >>> Computing Orderings...")
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.cpu().T.tolist())

        print("       - Computing Spectral Ordering (Fiedler)...")
        try:
            L = nx.normalized_laplacian_matrix(G)
            eigenvals, eigenvecs = sla.eigsh(L, k=2, which='SM')
            fiedler = eigenvecs[:, 1]
        except:
            fiedler = np.arange(num_nodes)

        self.perm = torch.tensor(np.argsort(fiedler), dtype=torch.long, device=device)
        self.inv_perm = torch.argsort(self.perm)

        pe_norm = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-6)
        self.pe = torch.tensor(pe_norm, dtype=torch.float32, device=device).view(-1, 1)

        print("       - Computing Random Ordering...")
        rng = np.random.default_rng(SEED)
        self.rand_perm = torch.tensor(rng.permutation(num_nodes), dtype=torch.long, device=device)
        self.inv_rand_perm = torch.argsort(self.rand_perm)

        print("       - Computing DeepWalk (Random Walk) Ordering...")
        A = nx.adjacency_matrix(G).todense()
        A = torch.tensor(A, dtype=torch.float32, device=device)
        d = A.sum(dim=1)
        D_inv = torch.diag(1.0 / torch.clamp(d, min=1e-5))

        P = D_inv @ A
        T = 10
        P_sum = P.clone()
        P_pow = P.clone()
        for _ in range(1, T):
            P_pow = P_pow @ P
            P_sum += P_pow
        P_sum = P_sum / T

        vol_G = d.sum()
        PMI = (vol_G / 1.0) * P_sum @ D_inv
        PMI = torch.log(torch.clamp(PMI, min=1.0))

        U, S, V = torch.pca_lowrank(PMI, q=1)
        pc1 = U[:, 0]
        self.dw_perm = torch.argsort(pc1)
        self.inv_dw_perm = torch.argsort(self.dw_perm)

    def process_batch(self, x, order_type='spectral'):
        if order_type == 'spectral':
            idx = self.perm
        elif order_type == 'deepwalk':
            idx = self.dw_perm
        else:
            idx = self.rand_perm
        return x[:, idx, :].permute(0, 2, 1)

    def recover_batch(self, x_seq, order_type='spectral'):
        x_seq = x_seq.permute(0, 2, 1)
        if order_type == 'spectral':
            idx = self.inv_perm
        elif order_type == 'deepwalk':
            idx = self.inv_dw_perm
        else:
            idx = self.inv_rand_perm
        return x_seq[:, idx, :]


class SpectralMambaNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.in_conv = nn.Conv1d(in_c, hid_c, 1)
        self.enc1 = nn.Sequential(nn.Conv1d(hid_c, hid_c, 9, padding=4, groups=hid_c), nn.GroupNorm(4, hid_c),
                                  nn.SiLU())
        self.down = nn.Conv1d(hid_c, hid_c * 2, 3, stride=2, padding=1)
        self.mid = nn.Sequential(nn.Conv1d(hid_c * 2, hid_c * 2, 13, padding=6, groups=hid_c * 2),
                                 nn.GroupNorm(8, hid_c * 2), nn.SiLU())
        self.up = nn.ConvTranspose1d(hid_c * 2, hid_c, 3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(nn.Conv1d(hid_c, hid_c, 9, padding=4, groups=hid_c), nn.SiLU())
        self.out_conv = nn.Conv1d(hid_c, out_c, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.enc1(x)
        x2 = self.down(x1)
        xm = self.mid(x2)
        xu = self.up(xm)
        if xu.size(2) != x1.size(2): xu = xu[:, :, :x1.size(2)]
        xu = xu + x1
        out = self.dec1(xu)
        return self.out_conv(out)


class ResGCNNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.norm1 = GraphNorm(hid_c)
        self.conv2 = GCNConv(hid_c, hid_c)
        self.norm2 = GraphNorm(hid_c)
        self.head = nn.Linear(hid_c + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index)))
        return self.head(torch.cat([x_in, x2], dim=-1))


class ResGATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads=4, concat=True)
        self.norm1 = GraphNorm(hid_c * 4)
        self.conv2 = GATConv(hid_c * 4, hid_c, heads=1, concat=True)
        self.norm2 = GraphNorm(hid_c)
        self.head = nn.Linear(hid_c + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x
        x1 = F.elu(self.norm1(self.conv1(x, edge_index)))
        x2 = F.elu(self.norm2(self.conv2(x1, edge_index)))
        return self.head(torch.cat([x_in, x2], dim=-1))


class SANNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, pe):
        super().__init__()
        self.pe = pe  # (N, 1)
        self.proj = nn.Linear(in_c + 1, hid_c)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_c, nhead=4, dim_feedforward=hid_c * 2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        x_pe = torch.cat([x, self.pe], dim=-1)
        x_h = self.proj(x_pe).unsqueeze(0)
        out = self.transformer(x_h).squeeze(0)
        return self.head(out)


class SGLinearAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        Q = F.relu(self.q(x))
        K = F.relu(self.k(x))
        V = self.v(x)
        KV = torch.matmul(K.t(), V)
        out = torch.matmul(Q, KV)
        normalizer = torch.matmul(Q, K.sum(dim=0, keepdim=True).t()) + 1e-6
        return out / normalizer


class SGFormerNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GCNConv(in_c, hid_c)
        self.att1 = SGLinearAttention(in_c, hid_c)
        self.norm1 = GraphNorm(hid_c)

        self.conv2 = GCNConv(hid_c, hid_c)
        self.att2 = SGLinearAttention(hid_c, hid_c)
        self.norm2 = GraphNorm(hid_c)

        self.conv3 = GCNConv(hid_c, hid_c)
        self.att3 = SGLinearAttention(hid_c, hid_c)
        self.norm3 = GraphNorm(hid_c)

        self.head = nn.Linear(hid_c * 3 + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x
        x1_g = self.conv1(x, edge_index)
        x1_a = self.att1(x)
        x1 = F.relu(self.norm1(x1_g + x1_a))

        x2_g = self.conv2(x1, edge_index)
        x2_a = self.att2(x1)
        x2 = F.relu(self.norm2(x2_g + x2_a)) + x1

        x3_g = self.conv3(x2, edge_index)
        x3_a = self.att3(x2)
        x3 = F.relu(self.norm3(x3_g + x3_a)) + x2

        return self.head(torch.cat([x_in, x1, x2, x3], dim=-1))


def compute_snr_single(pred, target):
    noise = torch.sum((pred - target) ** 2)
    sig = torch.sum(target ** 2)
    return 10 * torch.log10(sig / (noise + 1e-10)).item()


def compute_snr_batch(pred, target):
    noise = torch.sum((pred - target) ** 2, dim=(1, 2))
    sig = torch.sum(target ** 2, dim=(1, 2))
    return torch.mean(10 * torch.log10(sig / (noise + 1e-10))).item()


def run_experiment():
    try:
        loader = MNISTGraphLoader()
        X_noisy, X_clean, edge_index, num_nodes = loader.load()
    except Exception as e:
        print(f"Data Error: {e}")
        return

    input_snr = loader.input_snr_val

    n_train = 700
    n_val = 150
    train_x, train_y = X_noisy[:n_train], X_clean[:n_train]
    test_x, test_y = X_noisy[n_train + n_val:], X_clean[n_train + n_val:]

    print(f"Samples: Train={len(train_x)}, Test={len(test_x)}")

    serializer = SpectralSerializer()
    serializer.compute_ordering(edge_index, num_nodes)

    hid = 64

    m_gcn = ResGCNNet(1, hid, 1).to(device)
    m_gat = ResGATNet(1, hid, 1).to(device)
    m_rand = SpectralMambaNet(1, hid, 1).to(device)
    m_dw = SpectralMambaNet(1, hid, 1).to(device)
    m_spec = SpectralMambaNet(1, hid, 1).to(device)

    m_san = SANNet(1, hid, 1, serializer.pe).to(device)
    m_sgf = SGFormerNet(1, hid, 1).to(device)

    phase1_models = {
        "ResGCN": m_gcn,
        "ResGAT": m_gat,
        "Random-Mamba": m_rand,
        "DeepWalk-Mamba": m_dw,
        "Spectral-Mamba": m_spec
    }

    phase2_models = {
        "SAN": m_san,
        "SGFormer": m_sgf
    }

    final_stats = {}
    all_preds = {}
    epochs = 100

    print("\n========== Starting Experiments (Phase 1: Original Models) ==========")
    for name, model in phase1_models.items():
        print(f"Training {name}...")
        opt = torch.optim.AdamW(model.parameters(), lr=0.002)
        crit = nn.MSELoss()

        for ep in range(epochs):
            model.train()
            opt.zero_grad()

            batch_size = 32
            perm = torch.randperm(len(train_x))

            for i in range(0, len(train_x), batch_size):
                idx = perm[i:i + batch_size]
                bx, by = train_x[idx], train_y[idx]

                if "Mamba" in name:
                    if name == "Random-Mamba":
                        order = 'random'
                    elif name == "DeepWalk-Mamba":
                        order = 'deepwalk'
                    else:
                        order = 'spectral'
                    p = serializer.recover_batch(model(serializer.process_batch(bx, order)), order)
                else:
                    preds = [model(bx[j], edge_index) for j in range(len(bx))]
                    p = torch.stack(preds)

                loss = crit(p, by)
                loss.backward()
                opt.step()
                opt.zero_grad()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(test_x), 50):
                bx = test_x[i:i + 50]
                if "Mamba" in name:
                    if name == "Random-Mamba":
                        order = 'random'
                    elif name == "DeepWalk-Mamba":
                        order = 'deepwalk'
                    else:
                        order = 'spectral'
                    p = serializer.recover_batch(model(serializer.process_batch(bx, order)), order)
                else:
                    p = torch.stack([model(bx[j], edge_index) for j in range(len(bx))])
                preds.append(p)

            p_full = torch.cat(preds, dim=0)
            all_preds[name] = p_full
            snr = compute_snr_batch(p_full, test_y)
            mse = F.mse_loss(p_full, test_y).item()
            final_stats[name] = {'SNR': snr, 'MSE': mse}
            print(f">>> {name} Test SNR: {snr:.2f} dB")

    print("\n========== Starting Experiments (Phase 2: New SOTA Models) ==========")
    for name, model in phase2_models.items():
        print(f"Training {name}...")
        opt = torch.optim.AdamW(model.parameters(), lr=0.002)
        crit = nn.MSELoss()

        for ep in range(epochs):
            model.train()
            opt.zero_grad()

            batch_size = 32
            perm = torch.randperm(len(train_x))

            for i in range(0, len(train_x), batch_size):
                idx = perm[i:i + batch_size]
                bx, by = train_x[idx], train_y[idx]

                preds = [model(bx[j], edge_index) for j in range(len(bx))]
                p = torch.stack(preds)

                loss = crit(p, by)
                loss.backward()
                opt.step()
                opt.zero_grad()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(test_x), 50):
                bx = test_x[i:i + 50]
                p = torch.stack([model(bx[j], edge_index) for j in range(len(bx))])
                preds.append(p)

            p_full = torch.cat(preds, dim=0)
            all_preds[name] = p_full
            snr = compute_snr_batch(p_full, test_y)
            mse = F.mse_loss(p_full, test_y).item()
            final_stats[name] = {'SNR': snr, 'MSE': mse}
            print(f">>> {name} Test SNR: {snr:.2f} dB")

    display_order = ['ResGCN', 'ResGAT', 'SAN', 'SGFormer', 'Random-Mamba', 'DeepWalk-Mamba', 'Spectral-Mamba']

    print("\n========== Final Results (Avg) ==========")
    print(f"{'Method':<20} | {'Test SNR':<10}")
    print("-" * 35)
    for name in display_order:
        print(f"{name:<20} | {final_stats[name]['SNR']:<10.2f}")
    print("-" * 35)

    print("\nGenerating Paper-Ready Combined Visualization for samples [113, 99, 30]...")
    target_indices = [113, 99, 30]

    fig, axes = plt.subplots(3, 9, figsize=(22, 8))
    plt.subplots_adjust(wspace=0.01, hspace=0.05)

    display_names = {
        'Ground Truth': 'Ground Truth',
        'Input': f'Input ({input_snr:.1f}dB)',
        'ResGCN': 'ResGCN',
        'ResGAT': 'ResGAT',
        'SAN': 'SAN',
        'SGFormer': 'SGFormer',
        'Random-Mamba': 'Random-Mamba',
        'DeepWalk-Mamba': 'DeepWalk-Mamba',
        'Spectral-Mamba': 'Spectral-Mamba'
    }

    def show_img(ax, img_tensor, title=None, snr_val=None):
        img = img_tensor.view(IMG_SIZE, IMG_SIZE).cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=2.5)
        if snr_val is not None:
            ax.set_xlabel(f"{snr_val:.2f} dB", fontsize=11, color='blue')

    for row, idx in enumerate(target_indices):
        show_img(axes[row, 0], test_y[idx], title=display_names['Ground Truth'] if row == 0 else None)
        show_img(axes[row, 1], test_x[idx], title=display_names['Input'] if row == 0 else None)

        for col, m in enumerate(display_order):
            pred = all_preds[m][idx]
            s = compute_snr_single(pred, test_y[idx])
            show_img(axes[row, col + 2], pred, title=display_names[m] if row == 0 else None, snr_val=s)

        axes[row, 0].set_ylabel(f"Sample {idx}", fontsize=11, rotation=90, labelpad=2.5, fontweight='bold')

    save_path = "results_mnist_revisedR2/paper_combined_vis_with_all_sotas.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"√ Saved paper-ready figure to: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_experiment()