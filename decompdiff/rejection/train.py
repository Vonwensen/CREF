import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter

# 时间步嵌入函数
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# 优化后的模型结构（含跨注意力机制）
class EnhancedGNNClassifier(nn.Module):
    def __init__(self, ligand_dim=4, protein_dim=30, hidden_dim=128, num_layers=3, max_time_steps=1000):
        super().__init__()
        # 配体嵌入（3位置+1速度）
        self.ligand_embedding = nn.Sequential(
            nn.Linear(ligand_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 蛋白质嵌入（3位置+27特征）
        self.protein_embedding = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # 跨注意力层（增强配体-蛋白质交互）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 增强分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 融合配体和蛋白质全局特征
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim//2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 2)
        )
        
        self.hidden_dim = hidden_dim
        self.max_time_steps = max_time_steps

    def forward(self, ligand_state, protein_state, time):
        batch_size, n_ligand, _ = ligand_state.shape
        n_protein = protein_state.shape[1]
        
        # 提取位置信息计算相对距离
        ligand_pos = ligand_state[:, :, :3]
        protein_pos = protein_state[:, :, :3]
        
        ligand_expanded = ligand_pos.unsqueeze(2)
        protein_expanded = protein_pos.unsqueeze(1)
        rel_dist = torch.norm(ligand_expanded - protein_expanded, dim=-1)
        
        # 节点嵌入
        h_ligand = self.ligand_embedding(ligand_state)
        h_protein = self.protein_embedding(protein_state)
        
        # 时间步嵌入
        time_scaled = time / self.max_time_steps
        temb = timestep_embedding(time_scaled, self.hidden_dim)
        time_emb = self.time_embed(temb).unsqueeze(1)
        h_ligand = h_ligand + time_emb  # 融合时间信息
        
        # GNN层传播
        for layer in self.gnn_layers:
            h_expanded = h_ligand.unsqueeze(2).expand(-1, -1, n_protein, -1)
            p_expanded = h_protein.unsqueeze(1).expand(-1, n_ligand, -1, -1)
            dist_expanded = rel_dist.unsqueeze(-1)
            
            edge_input = torch.cat([h_expanded, p_expanded, dist_expanded], dim=-1)
            messages = layer(edge_input)
            h_ligand = h_ligand + messages.sum(dim=2)  # 残差连接
        
        # 跨注意力交互（配体为查询，蛋白质为键值）
        attn_output, _ = self.cross_attention(
            query=h_ligand,
            key=h_protein,
            value=h_protein
        )
        h_ligand = h_ligand + attn_output  # 残差连接
        
        # 融合全局特征
        h_global_ligand = h_ligand.mean(dim=1)
        h_global_protein = h_protein.mean(dim=1)
        h_global = torch.cat([h_global_ligand, h_global_protein], dim=-1)
        
        return self.classifier(h_global)


import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MultiTimeDiffusionDataset(Dataset):
    def __init__(self, labeled_dir, max_ligand_atoms=100, 
                 max_protein_atoms=500, normalize=True):
        self.data = []
        self.max_ligand_atoms = max_ligand_atoms
        self.max_protein_atoms = max_protein_atoms
        self.normalize = normalize
        
        # 加载数据并计算归一化参数
        self._load_data(labeled_dir)
        if normalize and len(self.data) > 0:
            self._compute_normalization_stats()
        else:
            self.ligand_mean = self.ligand_std = None
            self.protein_mean = self.protein_std = None

    def _load_data(self, labeled_dir):
        protein_files = [f for f in os.listdir(labeled_dir) 
                        if f.startswith('protein_') and f.endswith('.pt')]
        
        for file in protein_files:
            file_path = os.path.join(labeled_dir, file)
            try:
                ligand_list = torch.load(file_path)
                for ligand in ligand_list:
                    traj_length = len(ligand['ligand_pos_traj'])
                    if traj_length == 0:
                        continue
                    
                    # 处理完整配体轨迹
                    ligand_states = []
                    for t in range(traj_length):
                        ligand_pos = ligand['ligand_pos_traj'][t]
                        ligand_v = ligand['ligand_v_traj'][t]
                        if ligand_v.ndim == 1:
                            ligand_v = ligand_v.reshape(-1, 1)
                        ligand_state = np.concatenate([ligand_pos, ligand_v], axis=1)
                        ligand_states.append(ligand_state)
                    
                    # 处理蛋白质特征（假设蛋白质特征不随时间变化）
                    protein_pos = ligand['protein_pos']
                    protein_v = ligand['protein_v']
                    protein_state = np.concatenate([protein_pos, protein_v], axis=1)
                    
                    # 保留vina_score属性
                    self.data.append({
                        'ligand_traj': ligand_states,  # 完整轨迹列表
                        'protein_state': protein_state,
                        'traj_length': traj_length,    # 轨迹长度
                        'label': ligand['label'],
                        'vina_score': ligand.get('vina_score', None)  # 获取vina_score，默认None
                    })
            except Exception as e:
                print(f"Warning: 跳过错误文件 {file}: {e}")
                continue

    def _compute_normalization_stats(self):
        """计算特征均值和标准差用于归一化"""
        all_ligand = []
        all_protein = []
        
        for item in self.data:
            # 收集所有时间步的配体特征
            for ligand_state in item['ligand_traj']:
                all_ligand.append(ligand_state)
            # 收集蛋白质特征
            all_protein.append(item['protein_state'])
        
        # 计算配体特征的均值和标准差
        ligand_flat = np.concatenate(all_ligand, axis=0)
        self.ligand_mean = np.mean(ligand_flat, axis=0)
        self.ligand_std = np.std(ligand_flat, axis=0) + 1e-8  # 避免除零
        
        # 计算蛋白质特征的均值和标准差
        protein_flat = np.concatenate(all_protein, axis=0)
        self.protein_mean = np.mean(protein_flat, axis=0)
        self.protein_std = np.std(protein_flat, axis=0) + 1e-8

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 配体轨迹处理（3+1=4维）
        def process_ligand_traj(traj):
            processed_traj = []
            for state in traj:
                if state.ndim == 1:
                    state = state.reshape(-1, 4)
                if state.shape[1] != 4:
                    state = np.pad(state, ((0,0), (0, 4 - state.shape[1])), mode='constant')
                current_len = state.shape[0]
                if current_len < self.max_ligand_atoms:
                    padded = np.pad(state, ((0, self.max_ligand_atoms - current_len), (0,0)), mode='constant')
                else:
                    padded = state[:self.max_ligand_atoms]
                # 归一化
                if self.normalize and self.ligand_mean is not None:
                    padded = (padded - self.ligand_mean) / self.ligand_std
                processed_traj.append(padded)
            return np.stack(processed_traj, axis=0)  # 形状: (traj_length, max_ligand_atoms, 4)
        
        # 蛋白质处理（3+27=30维）
        def process_protein(arr):
            if arr.ndim == 1:
                arr = arr.reshape(-1, 30)
            if arr.shape[1] != 30:
                arr = np.pad(arr, ((0,0), (0, 30 - arr.shape[1])), mode='constant')
            current_len = arr.shape[0]
            if current_len < self.max_protein_atoms:
                arr = np.pad(arr, ((0, self.max_protein_atoms - current_len), (0,0)), mode='constant')
            else:
                arr = arr[:self.max_protein_atoms]
            # 归一化
            if self.normalize and self.protein_mean is not None:
                arr = (arr - self.protein_mean) / self.protein_std
            return arr
        
        ligand_traj = process_ligand_traj(sample['ligand_traj'])
        protein_state = process_protein(sample['protein_state'])
        
        return (
            torch.tensor(ligand_traj, dtype=torch.float32),  # 完整轨迹
            torch.tensor(protein_state, dtype=torch.float32),
            torch.tensor(sample['traj_length'], dtype=torch.long),  # 轨迹长度
            torch.tensor(sample['label'], dtype=torch.long),
            torch.tensor(sample['vina_score'], dtype=torch.float32) if sample['vina_score'] is not None else None  # vina_score
        )
    

# 改进训练函数
def train_model(model, dataloader, criterion, optimizer, scheduler, 
                num_epochs, device, save_path):
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # 解包batch数据（注意这里的batch结构与原Dataset对应）
            # ligand_traj: (batch_size, traj_length, max_ligand_atoms, 4)
            # protein_state: (batch_size, max_protein_atoms, 30)
            # traj_lengths: (batch_size,) 每个样本的实际轨迹长度
            # labels: (batch_size,)
            # vina_scores: 可选
            ligand_traj, protein_state, traj_lengths, labels, vina_scores = [x.to(device) if x is not None else None for x in batch]
            
            # 关键修改：为当前batch随机抽取一个统一的时间步t
            # 确保t在所有样本的有效轨迹长度范围内（取最小轨迹长度的-1作为上限）
            max_possible_t = torch.min(traj_lengths) - 1  # 所有样本都有的最大时间步
            if max_possible_t <= 0:
                t = torch.tensor(0, device=device)  # 处理轨迹长度为1的情况
            else:
                t = torch.randint(0, max_possible_t + 1, (1,), device=device).item()  # 随机抽取一个整数时间步
            
            # 提取当前时间步t的配体状态
            ligand_state = ligand_traj[:, t, :, :]  # 形状: (batch_size, max_ligand_atoms, 4)
            
            # 模型前向传播（time参数即为抽取的t）
            optimizer.zero_grad()
            output = model(ligand_state, protein_state, torch.tensor(t, device=device).repeat(labels.shape[0]))
            loss = criterion(output, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 学习率调度
        scheduler.step()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        print(f'Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # 保存最佳模型
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'最佳模型已保存（准确率 {best_acc:.2f}%）')



def main():
    parser = argparse.ArgumentParser(description='多时间步GNN分类器训练')
    parser.add_argument('--normalization_path', default=None, help='Output NPZ; defaults beside --save_path')
    parser.add_argument('--labeled_dir', default='proteins_with_labels', help='带标签数据目录')
    parser.add_argument('--num_epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率（建议5e-5）')
    parser.add_argument('--save_path', default='best_model.pt', help='模型保存路径')
    parser.add_argument('--max_time_steps', type=int, default=1000, help='最大时间步数量')
    # 采样参数
    parser.add_argument('--sample_strategy', default='random', 
                        choices=['random', 'interval', 'fixed'], help='时间步采样策略')
    parser.add_argument('--sample_ratio', type=float, default=0.001, 
                        help='随机/间隔采样比例（0-1）')
    parser.add_argument('--fixed_steps', type=int, default=5, 
                        help='固定策略保留的时间步数量')
    # 数据增强参数
    parser.add_argument('--no_normalize', action='store_true', 
                        help='不进行特征归一化')
    parser.add_argument('--weighted_loss', action='store_true', 
                        help='使用加权损失处理类别不平衡')
    parser.add_argument('--gpu', type=int, default=0, help='指定使用的CUDA设备编号（默认0）')
    
    args = parser.parse_args()
    
    # 根据参数设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"使用设备: CUDA {args.gpu} (设备名称: {torch.cuda.get_device_name(args.gpu)})")
    else:
        device = torch.device('cpu')
        print("CUDA不可用，使用设备: CPU")
    
    # 数据加载
    dataset = MultiTimeDiffusionDataset(
        args.labeled_dir, 
        max_ligand_atoms=100, 
        max_protein_atoms=500,

 

        normalize=not args.no_normalize
    )
    print(f"数据集大小: {len(dataset)} 样本")

    if len(dataset) == 0:
        raise ValueError("No usable labeled trajectories were loaded.")
    __import__('os').makedirs(__import__('os').path.dirname(__import__('os').path.abspath(args.save_path)), exist_ok=True)
    if not args.no_normalize:
        norm_path = args.normalization_path or __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(args.save_path)), 'normalization_params.npz')
        __import__('os').makedirs(__import__('os').path.dirname(__import__('os').path.abspath(norm_path)), exist_ok=True)
        np.savez(norm_path, ligand_mean=dataset.ligand_mean, ligand_std=dataset.ligand_std,
                 protein_mean=dataset.protein_mean, protein_std=dataset.protein_std)

    
    # 自定义数据拼接（适配完整轨迹数据）
    def custom_collate(batch):
        # 提取完整轨迹、蛋白质状态、轨迹长度、标签和vina_score
        ligand_traj = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)  # 形状: (batch_size, traj_length, max_ligand_atoms, 4)
        protein_state = torch.stack([item[1] for item in batch])  # 形状: (batch_size, max_protein_atoms, 30)
        traj_lengths = torch.stack([item[2] for item in batch])  # 形状: (batch_size,) 每个样本的实际轨迹长度
        labels = torch.stack([item[3] for item in batch])  # 形状: (batch_size,)
        
        # 处理vina_score（可能为None）
        vina_scores = [item[4] for item in batch]
        if any(vs is not None for vs in vina_scores):
            # 用nan填充None值
            vina_scores = torch.stack([
                vs if vs is not None else torch.tensor(float('nan')) 
                for vs in vina_scores
            ])
        else:
            vina_scores = None  # 全为None时返回None
        
        return (ligand_traj, protein_state, traj_lengths, labels, vina_scores)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        collate_fn=custom_collate  # 使用修改后的拼接函数
    )
    
    # 计算类别权重（处理不平衡）
    criterion = nn.CrossEntropyLoss()
    if args.weighted_loss and len(dataset) > 0:
        labels = [sample['label'] for sample in dataset.data]
        class_counts = Counter(labels)
        if len(class_counts) == 2:  # 确保有两个类别
            weight = torch.FloatTensor([
                len(labels)/class_counts[0], 
                len(labels)/class_counts[1]
            ]).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight)
            print(f"类别权重: {weight.cpu().numpy()}")
    
    # 模型初始化
    model = EnhancedGNNClassifier(
        ligand_dim=4,
        protein_dim=30,
        hidden_dim=128, 
        num_layers=5, 
        max_time_steps=args.max_time_steps
    )
    
    # 优化器和调度器（余弦退火）
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 开始训练
    train_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        device=device,
        save_path=args.save_path
    )
    
    print(f"训练完成，最佳模型已保存至 {args.save_path}")


if __name__ == "__main__":
    main()