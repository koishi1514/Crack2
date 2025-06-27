import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super(HaarWaveletTransform, self).__init__()

    def forward(self, x):
        """
        对输入的特征图进行Haar小波变换，分解为低频和高频分量
        x: 输入特征图, shape: (batch_size, channels, height, width)

        返回:
        - ll: 低频分量 (LL)
        - high_freq: 高频分量 (LH + HL + HH)
        """
        batch_size, channels, height, width = x.shape
        assert height % 2 == 0 and width % 2 == 0, "Height and width must be even for Haar transform"

        # Step 1: 垂直方向上的Haar变换
        # 使用卷积来实现 Haar 小波变换的平滑和差分
        low_pass = (x[:, :, ::2, :] + x[:, :, 1::2, :]) / 2  # 低频分量
        high_pass = (x[:, :, ::2, :] - x[:, :, 1::2, :]) / 2  # 高频分量

        # Step 2: 水平方向上的Haar变换
        ll = (low_pass[:, :, :, ::2] + low_pass[:, :, :, 1::2]) / 2  # 低频（LL）
        lh = (low_pass[:, :, :, ::2] - low_pass[:, :, :, 1::2]) / 2  # 水平高频（LH）
        hl = (high_pass[:, :, :, ::2] + high_pass[:, :, :, 1::2]) / 2  # 垂直高频（HL）
        hh = (high_pass[:, :, :, ::2] - high_pass[:, :, :, 1::2]) / 2  # 对角高频（HH）

        # 高频分量是LH + HL + HH
        high_freq = lh + hl + hh
        low_freq = ll

        return low_freq, high_freq

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        """
        初始化 Haar 小波注意力模块。
        Args:
            in_channels (int): 输入特征图的通道数。
            reduction_ratio (int): 用于线性层中通道数缩减的比例。
        """
        super(HaarWavelet, self).__init__()
        self.haar_transform = HaarWaveletTransform()
        self.in_channels = in_channels

        # 小型神经网络的两个分支
        # 分支 1: 全局平均池化 + 两个线性层
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，输出 (B, C, 1, 1)
            nn.Flatten(),             # 展平为 (B, C)
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )

        # 分支 2: 两个线性层 (使用 1x1 卷积实现)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)。
                              注意 height 和 width 必须是偶数。
        Returns:
            torch.Tensor: 经过 Haar 小波变换和注意力加权后的高频分量。
                          形状为 (batch_size, channels, height/2, width/2)。
        """
        low_f, high_f = self.haar_transform(x)
        fused_input = low_f + high_f

        # fused_input = nn.AdaptiveAvgPool2d(1)(fused_input)  # (B, C, 1, 1)
        # fused_input = fused_input.squeeze(-1).squeeze(-1)  # (B, C)
        # fused_input = nn.Linear(self.in_channels, self.in_channels//2)(fused_input)  # (B, C)


        branch1_out = self.branch1(fused_input)  # (B, C)
        branch1_out = branch1_out.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        branch2_out = self.branch2(fused_input)  # (B, C, H/2, W/2)
        attention_weights = self.sigmoid(branch1_out + branch2_out)


        output = high_f * attention_weights + low_f

        return output

class Haar_CrossAttention(nn.Module):
    def __init__(self, in_channels, in_channels_q, d_model, num_heads=1):
        """
        初始化 Haar 小波交叉注意力模块。
        Args:
            in_channels_haar_input (int): 用于生成 Query 的 Haar 输入特征图的通道数。
            in_channels_encoder_output (int): 来自 U-Net 编码器输出 (K, V) 的特征图通道数。
            d_model (int): Query, Key, Value 投影后的维度。
            num_heads (int): 多头注意力的头数。
        """
        super(Haar_CrossAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Query 投影层 (来自 Haar 变换的高频分量)
        # Haar 变换的输出通道数与输入通道数相同
        self.query_proj = nn.Conv2d(in_channels_q, d_model, kernel_size=1)

        # Key 和 Value 投影层 (来自 U-Net 编码器输出)
        self.key_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.value_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

        # 最终输出投影层，将 d_model 映射回 encoder_output 的通道数
        self.output_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_haar_input, x_encoder_output):

        B, C, H, W = x_encoder_output.shape


        # 投影 Q, K, V 到 d_model 维度
        q = self.query_proj(x_haar_input) # (B, d_model, target_H, target_W)
        k = self.key_proj(x_encoder_output) # (B, d_model, target_H, target_W)
        v = self.value_proj(x_encoder_output) # (B, d_model, target_H, target_W)

        # 准备多头注意力：将通道维度 d_model 分割为 num_heads * head_dim
        # 并将空间维度展平
        # (B, d_model, H, W) -> (B, num_heads, head_dim, H*W)
        q = q.view(B, self.num_heads, self.head_dim, -1) # -1 自动计算为 H*W
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)

        # 5. 转置维度以便进行批处理矩阵乘法 (B, num_heads, H*W, head_dim)
        q = q.permute(0, 1, 3, 2) # (B, heads, N_q, head_dim)
        k = k.permute(0, 1, 3, 2) # (B, heads, N_kv, head_dim)
        v = v.permute(0, 1, 3, 2) # (B, heads, N_kv, head_dim)

        # 6. 计算注意力分数 (Scaled Dot-Product Attention)
        # (B, heads, N_q, head_dim) @ (B, heads, head_dim, N_kv) -> (B, heads, N_q, N_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = self.softmax(scores)

        # 7. 应用注意力到 Value
        # (B, heads, N_q, N_kv) @ (B, heads, N_kv, head_dim) -> (B, heads, N_q, head_dim)
        output = torch.matmul(attention_weights, v)

        # 8. 拼接多头并重塑回空间维度
        # (B, heads, N_q, head_dim) -> (B, N_q, heads * head_dim) -> (B, d_model, H, W)
        output = output.permute(0, 2, 1, 3).contiguous().view(B, self.d_model, H, W)

        # 9. 最终投影到原始 encoder_output 的通道数
        output = self.output_proj(output) # (B, C_enc, H, W)


        return output

if __name__ == "__main__":
    # 假设输入特征图为 32x32，通道数为 64
    input_tensor = torch.randn(1, 64, 32, 32) # Batch_size=2, Channels=64, Height=32, Width=32

    # 实例化模块
    haar_attention_module = HaarWavelet(in_channels=64, reduction_ratio=2)

    # 前向传播
    output_tensor = haar_attention_module(input_tensor)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Output tensor shape: {output_tensor.shape}") # 预期 (2, 64, 16, 16)
