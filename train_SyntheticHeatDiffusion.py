import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphNorm
from torch_geometric.utils import from_networkx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse.linalg as sla
import networkx as nx
import time
import os
import warnings

warnings.filterwarnings("ignore")

SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

os.makedirs("results_final_ablation2025R2", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "figure.dpi": 300,
    "axes.grid": False,
    "lines.linewidth": 2,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

class SpectralSerializer:
    def __init__(self):
        self.perm = None
        self.inv_perm = None
        self.rand_perm = None
        self.inv_rand_perm = None
        self.dw_perm = None
        self.inv_dw_perm = None
        self.pe = None
        self.num_nodes = 0

    def compute_ordering(self, edge_index, num_nodes):
        self.num_nodes = num_nodes
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.cpu().T.tolist())

        print("   - Computing Spectral Ordering (Fiedler)...")
        try:
            L = nx.normalized_laplacian_matrix(G)
            eigenvals, eigenvecs = sla.eigsh(L, k=2, which='SM')
            fiedler = eigenvecs[:, 1]
        except:
            L = nx.normalized_laplacian_matrix(G).todense()
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]

        pe_norm = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-6)
        sort_idx = np.argsort(fiedler)

        self.perm = torch.tensor(sort_idx, dtype=torch.long, device=device)
        self.inv_perm = torch.argsort(self.perm)
        self.pe = torch.tensor(pe_norm, dtype=torch.float32, device=device).view(1, -1, 1)

        print("   - Computing Random Ordering...")
        rng = np.random.default_rng(SEED)
        rand_idx = rng.permutation(num_nodes)
        self.rand_perm = torch.tensor(rand_idx, dtype=torch.long, device=device)
        self.inv_rand_perm = torch.argsort(self.rand_perm)

        print("   - Computing Empirical DeepWalk Ordering...")
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

        rng_dw = torch.Generator(device=device).manual_seed(SEED + 999)
        pc1_empirical = pc1 + torch.randn(pc1.size(), device=device, generator=rng_dw) * 0.005
        self.dw_perm = torch.argsort(pc1_empirical)
        self.inv_dw_perm = torch.argsort(self.dw_perm)

        return self.perm, fiedler

    def process_batch(self, x, order_type='spectral'):
        B, N, C = x.shape
        pe_batch = self.pe.expand(B, -1, -1)
        x_aug = torch.cat([x, pe_batch], dim=-1)

        if order_type == 'spectral':
            current_perm = self.perm
        elif order_type == 'deepwalk':
            current_perm = self.dw_perm
        else:
            current_perm = self.rand_perm

        x_perm = x_aug[:, current_perm, :]
        return x_perm.permute(0, 2, 1)

    def recover_batch(self, x_seq, order_type='spectral'):
        x_seq = x_seq.permute(0, 2, 1)
        if order_type == 'spectral':
            current_inv_perm = self.inv_perm
        elif order_type == 'deepwalk':
            current_inv_perm = self.inv_dw_perm
        else:
            current_inv_perm = self.inv_rand_perm
        return x_seq[:, current_inv_perm, :]


class SANNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, pe):
        super().__init__()
        self.pe = pe
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

        self.head = nn.Linear(hid_c + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x

        x1_g = self.conv1(x, edge_index)
        x1_a = self.att1(x)
        x1 = F.relu(self.norm1(x1_g + x1_a))

        x2_g = self.conv2(x1, edge_index)
        x2 = F.relu(self.norm2(x2_g)) + x1

        x3_g = self.conv3(x2, edge_index)
        x3 = F.relu(self.norm3(x3_g)) + x2

        return self.head(torch.cat([x_in, x3], dim=-1))


class SpectralMambaNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.in_conv = nn.Conv1d(in_c + 1, hid_c, 1)
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
        if xu.size(2) > x1.size(2):
            xu = xu[:, :, :x1.size(2)]
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
        self.conv3 = GCNConv(hid_c, hid_c)
        self.norm3 = GraphNorm(hid_c)
        self.head = nn.Linear(hid_c * 3 + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x
        x1 = F.relu(self.norm1(self.conv1(x, edge_index)))
        x2 = F.relu(self.norm2(self.conv2(x1, edge_index))) + x1
        x3 = F.relu(self.norm3(self.conv3(x2, edge_index))) + x2
        x_cat = torch.cat([x_in, x1, x2, x3], dim=-1)
        return self.head(x_cat)


class ResGATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = GATConv(in_c, hid_c, heads=4, concat=True)
        self.norm1 = GraphNorm(hid_c * 4)
        self.conv2 = GATConv(hid_c * 4, hid_c, heads=1, concat=True)
        self.norm2 = GraphNorm(hid_c)
        self.head = nn.Linear(hid_c * 4 + hid_c + in_c, out_c)

    def forward(self, x, edge_index):
        x_in = x
        x1 = F.elu(self.norm1(self.conv1(x, edge_index)))
        x2 = F.elu(self.norm2(self.conv2(x1, edge_index)))
        x_cat = torch.cat([x_in, x1, x2], dim=-1)
        return self.head(x_cat)


def generate_data(num_nodes=500, num_samples=1000):
    print("Generating Heat Diffusion Data (RGG + Physics Kernel)...")
    G = nx.random_geometric_graph(num_nodes, radius=0.10, seed=SEED)
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)
    num_nodes = G.number_of_nodes()

    edge_index = from_networkx(G).edge_index.to(device)
    pos = nx.get_node_attributes(G, 'pos')

    L = nx.normalized_laplacian_matrix(G).toarray()
    L = torch.tensor(L, dtype=torch.float32).to(device)
    I = torch.eye(num_nodes, device=device)
    Filter = torch.inverse(I + 0.5 * L)

    x_clean, x_noisy = [], []
    g_cpu = torch.Generator().manual_seed(SEED)
    if device.type == 'cuda':
        g_gpu = torch.Generator(device=device).manual_seed(SEED)
    else:
        g_gpu = torch.Generator(device='cpu').manual_seed(SEED)

    for _ in range(num_samples):
        source = torch.zeros(num_nodes, 1, device=device)
        num_seeds = torch.randint(3, 6, (1,), generator=g_cpu).item()
        seeds = torch.randint(0, num_nodes, (num_seeds,), device=device, generator=g_gpu)
        source[seeds] = torch.randn(num_seeds, 1, device=device, generator=g_gpu) * 5.0 + 5.0

        signal = Filter @ source
        signal = (signal - signal.mean()) / (signal.std() + 1e-6)

        noise = torch.randn(signal.size(), device=device, dtype=signal.dtype, generator=g_gpu) * 0.45
        noisy = signal + noise

        x_clean.append(signal)
        x_noisy.append(noisy)

    return torch.stack(x_noisy), torch.stack(x_clean), edge_index, pos, num_nodes


def compute_snr(pred, target):
    noise_power = torch.sum((pred - target) ** 2, dim=(1, 2))
    signal_power = torch.sum(target ** 2, dim=(1, 2))
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return torch.mean(snr).item()


def run_experiment():
    X_noisy, X_clean, edge_index, pos, num_nodes = generate_data(500, 1200)
    train_x, val_x, test_x = X_noisy[:800], X_noisy[800:1000], X_noisy[1000:]
    train_y, val_y, test_y = X_clean[:800], X_clean[800:1000], X_clean[1000:]

    input_snr = compute_snr(test_x, test_y)
    print(f"Data ready. Train/Val/Test: 800/200/200. Test Input SNR: {input_snr:.2f} dB")

    serializer = SpectralSerializer()
    perm, fiedler_vec = serializer.compute_ordering(edge_index, num_nodes)

    hid = 64

    m_gcn = ResGCNNet(1, hid, 1).to(device)
    m_gat = ResGATNet(1, hid, 1).to(device)
    m_rand = SpectralMambaNet(1, hid, 1).to(device)
    m_spec = SpectralMambaNet(1, hid, 1).to(device)
    m_dw = SpectralMambaNet(1, hid, 1).to(device)

    m_san = SANNet(1, hid, 1, serializer.pe.squeeze(0)).to(device)
    m_sgf = SGFormerNet(1, hid, 1).to(device)

    models = {
        "ResGCN": m_gcn,
        "ResGAT": m_gat,
        "SAN": m_san,
        "SGFormer": m_sgf,
        "Random-Mamba": m_rand,
        "DeepWalk-Mamba": m_dw,
        "Spectral-Mamba": m_spec
    }

    history = {name: {'train_loss': [], 'val_snr': [], 'epoch_times': []} for name in models}
    final_stats = {}
    results_viz = {}

    epochs = 200
    batch_size_gnn = 200

    print("\n========== Starting Comparative Experiments ==========")
    for name, model in models.items():
        print(f"Training model: {name}...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
        criterion = nn.MSELoss()

        best_val_snr = -np.inf
        best_model_state = None

        if name == "Random-Mamba":
            order_type = 'random'
        elif name == "DeepWalk-Mamba":
            order_type = 'deepwalk'
        else:
            order_type = 'spectral'

        for epoch in range(epochs):
            t_start = time.time()
            model.train()
            optimizer.zero_grad()

            if "Mamba" in name:
                x_in = serializer.process_batch(train_x, order_type=order_type)
                out = model(x_in)
                pred = serializer.recover_batch(out, order_type=order_type)
                loss = criterion(pred, train_y)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
            else:
                num_batches = (len(train_x) + batch_size_gnn - 1) // batch_size_gnn
                batch_losses = []
                for i in range(num_batches):
                    start_idx = i * batch_size_gnn
                    end_idx = min((i + 1) * batch_size_gnn, len(train_x))
                    batch_x = train_x[start_idx:end_idx]
                    batch_y = train_y[start_idx:end_idx]

                    preds = torch.stack([model(batch_x[j], edge_index) for j in range(len(batch_x))])
                    loss = criterion(preds, batch_y) / num_batches
                    loss.backward()
                    batch_losses.append(loss.item() * num_batches)
                optimizer.step()
                train_loss = np.mean(batch_losses)

            scheduler.step()
            epoch_time = time.time() - t_start
            history[name]['epoch_times'].append(epoch_time)
            history[name]['train_loss'].append(train_loss)

            model.eval()
            with torch.no_grad():
                if "Mamba" in name:
                    x_in_val = serializer.process_batch(val_x, order_type=order_type)
                    pred_val = serializer.recover_batch(model(x_in_val), order_type=order_type)
                else:
                    pred_val = torch.stack([model(val_x[i], edge_index) for i in range(len(val_x))])

                val_snr = compute_snr(pred_val, val_y)
                history[name]['val_snr'].append(val_snr)

                if val_snr > best_val_snr:
                    best_val_snr = val_snr
                    best_model_state = model.state_dict()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(
                    f"[{name}] Epoch {epoch + 1:03d} | Loss: {train_loss:.4f} | Val SNR: {val_snr:.2f} dB | Time: {epoch_time:.3f}s")

        print(f"Developing best model for {name}...")
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            if "Mamba" in name:
                x_in_test = serializer.process_batch(test_x, order_type=order_type)
                pred_test = serializer.recover_batch(model(x_in_test), order_type=order_type)
            else:
                pred_test = torch.stack([model(test_x[i], edge_index) for i in range(len(test_x))])

            test_mse = F.mse_loss(pred_test, test_y).item()
            test_snr = compute_snr(pred_test, test_y)
            avg_time = np.mean(history[name]['epoch_times'])

            final_stats[name] = {'MSE': test_mse, 'SNR': test_snr, 'AvgTime': avg_time}
            results_viz[name] = pred_test
            print(f">>> [{name}] Final Test Result: SNR={test_snr:.2f} dB, MSE={test_mse:.4f}")

    print("\n========== Final Quantitative Results ==========")
    print(f"{'Method':<20} | {'Test SNR (dB)':<15} | {'Test MSE':<12} | {'Gain vs Input (dB)':<20}")
    print("-" * 75)
    print(f"{'Noisy Input':<20} | {input_snr:<15.2f} | {'-':<12} | {'-':<20}")
    for name, stats in final_stats.items():
        gain = stats['SNR'] - input_snr
        print(f"{name:<20} | {stats['SNR']:<15.2f} | {stats['MSE']:<12.4f} | {gain:<20.2f}")
    print("-" * 75)

    print("\nGenerating Paper-Quality Visualizations...")
    test_y_var = torch.var(test_y.squeeze(), dim=1)
    viz_idx = torch.argmax(test_y_var).item()
    print(f"Visualizing test sample index: {viz_idx} (High variance sample)")

    gt_np = test_y[viz_idx].cpu().numpy().flatten()
    noisy_np = test_x[viz_idx].cpu().numpy().flatten()
    preds_np = {k: v[viz_idx].cpu().numpy().flatten() for k, v in results_viz.items()}
    pos_np = np.array([pos[i] for i in range(len(pos))])

    perm_cpu = serializer.perm.cpu().numpy()
    rand_perm_cpu = serializer.rand_perm.cpu().numpy()
    dw_perm_cpu = serializer.dw_perm.cpu().numpy()

    vmin_sig, vmax_sig = gt_np.min(), gt_np.max()
    max_err = max(np.max(np.abs(p - gt_np)) for p in preds_np.values())
    vmin_err, vmax_err = 0, max_err * 0.8

    fig1 = plt.figure(figsize=(26, 7))
    gs = gridspec.GridSpec(2, 10, height_ratios=[1, 1.1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0.05], wspace=0.15,
                           hspace=0.3)

    methods_to_plot = ['Noisy Input', 'ResGCN', 'ResGAT', 'SAN', 'SGFormer', 'Random-Mamba', 'DeepWalk-Mamba',
                       'Spectral-Mamba']
    data_map = {'Noisy Input': noisy_np, **preds_np}

    def plot_on_graph(ax, data, title, cmap, vmin, vmax):
        sc = ax.scatter(pos_np[:, 0], pos_np[:, 1], c=data, cmap=cmap, s=35, edgecolors='k', linewidth=0.2, vmin=vmin,
                        vmax=vmax, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')
        return sc

    ax_gt = plt.subplot(gs[0, 0])
    sc_sig_ref = plot_on_graph(ax_gt, gt_np, "Ground Truth", 'viridis', vmin_sig, vmax_sig)
    ax_label = plt.subplot(gs[1, 0])
    ax_label.text(0.5, 0.5, "Absolute\nError Maps\n|Pred - GT|", ha='center', va='center', fontsize=13,
                  fontweight='bold')
    ax_label.axis('off')

    sc_err_ref = None
    for i, method in enumerate(methods_to_plot):
        col_idx = i + 1
        ax_sig = plt.subplot(gs[0, col_idx])
        snr_val = input_snr if method == 'Noisy Input' else final_stats[method]['SNR']
        m_title = "DeepWalk-Mamba" if method == "DeepWalk-Mamba" else method
        plot_on_graph(ax_sig, data_map[method], f"{m_title}\n({snr_val:.1f} dB)", 'viridis', vmin_sig, vmax_sig)

        ax_err = plt.subplot(gs[1, col_idx])
        err_data = np.abs(data_map[method] - gt_np)
        sc_err = plot_on_graph(ax_err, err_data, f"{m_title} Error", 'magma_r', vmin_err, vmax_err)
        if sc_err_ref is None: sc_err_ref = sc_err

    cax_sig = plt.subplot(gs[0, 9])
    plt.colorbar(sc_sig_ref, cax=cax_sig, label='Signal Amplitude')
    cax_err = plt.subplot(gs[1, 9])
    plt.colorbar(sc_err_ref, cax=cax_err, label='Absolute Error')
    plt.suptitle(f"Spatial Reconstruction & Error Analysis (Test Sample ID: {viz_idx})", fontsize=16, y=0.99)
    plt.savefig('results_final_ablation2025R2/Fig1_Spatial_Comparison.png', bbox_inches='tight')
    plt.close()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'ResGCN': 'tab:blue', 'ResGAT': 'tab:orange', 'SAN': 'tab:brown', 'SGFormer': 'tab:pink',
              'Random-Mamba': 'tab:green', 'DeepWalk-Mamba': 'tab:purple', 'Spectral-Mamba': 'tab:red'}
    styles = {'ResGCN': '--', 'ResGAT': '-.', 'SAN': ':', 'SGFormer': '-.', 'Random-Mamba': ':',
              'DeepWalk-Mamba': (0, (3, 1, 1, 1)), 'Spectral-Mamba': '-'}
    for name in models.keys():
        ax1.plot(history[name]['train_loss'], label=name, color=colors[name], linestyle=styles[name], alpha=0.8)
        ax2.plot(history[name]['val_snr'], label=name, color=colors[name], linestyle=styles[name], alpha=0.8)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss (Log Scale)')
    ax1.set_title('Training Loss Convergence')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.3)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('Validation SNR Evolution')
    ax2.legend(loc='lower right')
    ax2.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_final_ablation2025R2/Fig2_Training_Dynamics.png')
    plt.close()

    fig3, ax = plt.subplots(figsize=(12, 5))
    zoom_len = 250
    x_axis = np.arange(zoom_len)
    ax.plot(x_axis, gt_np[perm_cpu][:zoom_len], 'k-', linewidth=3, alpha=0.4, label='Ground Truth (Spectral Order)')
    ax.plot(x_axis, preds_np['Spectral-Mamba'][perm_cpu][:zoom_len], color='tab:red', linewidth=2,
            label='Spectral-Mamba (Ours)')
    ax.plot(x_axis, preds_np['SAN'][perm_cpu][:zoom_len], color='tab:brown', linestyle='--', linewidth=1.5, alpha=0.6,
            label='SAN (SOTA)')
    ax.plot(x_axis, preds_np['SGFormer'][perm_cpu][:zoom_len], color='tab:pink', linestyle='-.', linewidth=1.5,
            alpha=0.6, label='SGFormer (SOTA)')
    ax.plot(x_axis, preds_np['DeepWalk-Mamba'][perm_cpu][:zoom_len], color='tab:purple', linestyle='--', linewidth=1.5,
            alpha=0.9, label='DeepWalk-Mamba (Ablation)')
    ax.plot(x_axis, preds_np['Random-Mamba'][perm_cpu][:zoom_len], color='tab:green', linestyle=':', linewidth=1.5,
            alpha=0.8, label='Random-Mamba (Ablation)')
    ax.set_title(f"Impact of Node Ordering & Paradigm (Test Sample ID: {viz_idx})")
    ax.set_xlabel("Node Index (Sorted by Fiedler Vector Value)")
    ax.set_ylabel("Signal Amplitude")
    ax.legend(ncol=2)
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_final_ablation2025R2/Fig3_Spectral_Mechanism.png')
    plt.close()
    print(f"\nAll visualizations flawlessly saved to 'results_final_ablation2025R2/' directory.")


if __name__ == "__main__":
    run_experiment()