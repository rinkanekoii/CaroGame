from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import torch
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import onnxruntime as ort

_COORD_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

class ONNXNetWrapper(nn.Module):
    def __init__(self, onnx_path: str):
        super().__init__()
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(
            onnx_path, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.value_name = self.session.get_outputs()[0].name
        self.policy_name = self.session.get_outputs()[1].name
        self.use_coords = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        TARGET = 15
        if H != TARGET or W != TARGET:
            padded_x = torch.zeros((B, C, TARGET, TARGET), dtype=x.dtype, device=x.device)
            start_h = max(0, (TARGET - H) // 2)
            start_w = max(0, (TARGET - W) // 2)
            end_h = start_h + min(H, TARGET)
            end_w = start_w + min(W, TARGET)
            
            src_start_h = max(0, (H - TARGET) // 2)
            src_start_w = max(0, (W - TARGET) // 2)
            src_end_h = src_start_h + min(H, TARGET)
            src_end_w = src_start_w + min(W, TARGET)
            
            padded_x[:, :, start_h:end_h, start_w:end_w] = x[:, :, src_start_h:src_end_h, src_start_w:src_end_w]
            x_for_onnx = padded_x
        else:
            x_for_onnx = x

        x_np = x_for_onnx.cpu().numpy()
        expected_type = self.session.get_inputs()[0].type
        if expected_type == 'tensor(float16)':
            x_np = x_np.astype(np.float16)
            
        # Process each sample individually to avoid ONNX Reshape batch issues
        if B > 1:
            value_parts = []
            policy_parts = []
            for j in range(B):
                v_j, p_j = self.session.run(
                    [self.value_name, self.policy_name], {self.input_name: x_np[j:j+1]}
                )
                value_parts.append(v_j)
                policy_parts.append(p_j)
            value_out = np.concatenate(value_parts, axis=0)
            policy_out = np.concatenate(policy_parts, axis=0)
        else:
            value_out, policy_out = self.session.run(
                [self.value_name, self.policy_name], {self.input_name: x_np}
            )
        policy_t = torch.from_numpy(policy_out).to(x.device)
        value_t = torch.from_numpy(value_out).to(x.device)
        
        if H != TARGET or W != TARGET:
            policy_2d = policy_t.view(B, TARGET, TARGET)
            out_policy = torch.full((B, H, W), -10000.0, dtype=policy_t.dtype, device=x.device)
            out_policy[:, src_start_h:src_end_h, src_start_w:src_end_w] = policy_2d[:, start_h:end_h, start_w:end_w]
            policy_t = out_policy.reshape(B, H * W)

        return value_t, policy_t


def check_win_adaptive(board: np.ndarray, row: int, col: int, player: int, win_length: int = 5, rule_type: int = 0) -> bool:
    board_rows, board_cols = board.shape
    actual_win_length = min(win_length, max(3, min(board_rows, board_cols) - 2))
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1
        blocks = 0
        
        # Check one direction
        for i in range(1, actual_win_length + 1):
            rr, cc = row + dx * i, col + dy * i
            if 0 <= rr < board_rows and 0 <= cc < board_cols:
                if board[rr, cc] == player:
                    count += 1
                elif board[rr, cc] == -player:
                    blocks += 1
                    break
                else:
                    break
            else:
                blocks += 1
                break
                
        # Check opposite direction
        for i in range(1, actual_win_length + 1):
            rr, cc = row - dx * i, col - dy * i
            if 0 <= rr < board_rows and 0 <= cc < board_cols:
                if board[rr, cc] == player:
                    count += 1
                elif board[rr, cc] == -player:
                    blocks += 1
                    break
                else:
                    break
            else:
                blocks += 1
                break
                
        if count >= actual_win_length:
            if rule_type == 1 and count == actual_win_length and blocks == 2:
                continue
            return True
    return False


def is_useless_move(board: np.ndarray, row: int, col: int, player: int, win_length: int = 5, rule_type: int = 0) -> bool:
    """Check if placing a stone creates a 5-in-a-row that is blocked on both ends (a wasted move)."""
    if rule_type != 1:
        return False
        
    board_rows, board_cols = board.shape
    actual_win_length = min(win_length, max(3, min(board_rows, board_cols) - 2))
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for dx, dy in directions:
        count = 1
        blocks = 0
        
        # Check one direction
        for i in range(1, actual_win_length + 1):
            rr, cc = row + dx * i, col + dy * i
            if 0 <= rr < board_rows and 0 <= cc < board_cols:
                if board[rr, cc] == player: count += 1
                elif board[rr, cc] == -player: blocks += 1; break
                else: break
            else: blocks += 1; break
                
        # Check opposite direction
        for i in range(1, actual_win_length + 1):
            rr, cc = row - dx * i, col - dy * i
            if 0 <= rr < board_rows and 0 <= cc < board_cols:
                if board[rr, cc] == player: count += 1
                elif board[rr, cc] == -player: blocks += 1; break
                else: break
            else: blocks += 1; break
                
        if count == actual_win_length and blocks == 2:
            return True
            
    return False


def state_to_tensor(board: np.ndarray, player: int, use_coords: bool = False, rule_type: int = 0) -> np.ndarray:
    board_rows, board_cols = board.shape
    b0 = (board == player).astype(np.float32)
    b1 = (board == -player).astype(np.float32)
    b2 = (board == 0).astype(np.float32)
    b3 = np.full((board_rows, board_cols), float(rule_type), dtype=np.float32)
    planes = [b0, b1, b2, b3]
    if use_coords:
        cached = _COORD_CACHE.get((board_rows, board_cols))
        if cached is None:
            r_coords = np.linspace(-1.0, 1.0, board_rows, dtype=np.float32)[:, None] * np.ones((1, board_cols), dtype=np.float32)
            c_coords = np.linspace(-1.0, 1.0, board_cols, dtype=np.float32)[None, :] * np.ones((board_rows, 1), dtype=np.float32)
            _COORD_CACHE[(board_rows, board_cols)] = (r_coords, c_coords)
        else:
            r_coords, c_coords = cached
        planes.append(r_coords)
        planes.append(c_coords)
    return np.stack(planes, axis=0)


def filter_edge_moves(board: np.ndarray, moves: Dict[Tuple[int, int], float], margin: int = 2) -> Dict[Tuple[int, int], float]:
    """Keep moves that are near existing pieces (radius 6) or have high neural net prior.
    
    The expanded radius-6 neighborhood allows the AI to 'think outside the box' and
    plan moves that start a new cluster far from the current battle.
    We also retain up to 10% of candidate moves from the top neural-net predictions
    that are not already near a piece, giving the AI a chance to discover long-range
    strategies it learned during training.
    """
    board_rows, board_cols = board.shape
    if not moves:
        return moves
    
    # Build a fast boolean occupancy mask
    near_mask = np.zeros((board_rows, board_cols), dtype=bool)
    
    # Add margin interior (never filter out center cells)
    if margin > 0 and 2 * margin < board_rows:
        near_mask[margin:board_rows - margin, margin:board_cols - margin] = True
    
    # Expand radius to 6 around every occupied cell
    occupied_positions = list(zip(*np.where(board != 0)))
    for r, c in occupied_positions:
        r_lo = max(0, r - 6)
        r_hi = min(board_rows, r + 7)
        c_lo = max(0, c - 6)
        c_hi = min(board_cols, c + 7)
        near_mask[r_lo:r_hi, c_lo:c_hi] = True
    
    # Primary filter: moves near pieces or in center
    filtered = {mv: prob for mv, prob in moves.items() if near_mask[mv[0], mv[1]]}
    
    # Secondary: add top far-field moves from neural net (up to 5% of total, min 3)
    # This lets the AI occasionally open a new front on the board
    if len(moves) > 0:
        far_moves = {mv: prob for mv, prob in moves.items() if not near_mask[mv[0], mv[1]]}
        if far_moves:
            num_far_to_keep = max(3, len(moves) // 20)  # keep top 5%
            top_far = sorted(far_moves.items(), key=lambda x: x[1], reverse=True)[:num_far_to_keep]
            for mv, prob in top_far:
                filtered[mv] = prob
    
    if not filtered:
        return moves
    
    total = sum(filtered.values())
    if total <= 0:
        return moves
    return {key: value / total for key, value in filtered.items()}


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3))
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(y.size(0), y.size(1), 1, 1)


class PreNormResBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_path = drop_path
        self.se = SEBlock(channels, reduction=8)
        nn.init.constant_(self.conv2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_path > 0:
            if torch.rand(1).item() < self.drop_path:
                return x
        residual = x
        out = torch.relu(self.norm1(x))
        out = self.conv1(out)
        out = torch.relu(self.norm2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        return residual + out


class GlobalContextBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(16, channels // reduction)
        self.context = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Softmax(dim=2))
        self.transform = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        ctx = self.context(x)
        ctx = ctx.view(batch, 1, height * width)
        x_flat = x.view(batch, channels, height * width)
        global_ctx = torch.bmm(x_flat, ctx.transpose(1, 2))
        global_ctx = global_ctx.view(batch, channels, 1, 1)
        attn = self.transform(global_ctx)
        return x * attn


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_flat = x.view(batch, channels, height * width).permute(0, 2, 1)
        normed = self.norm1(x_flat)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.permute(0, 2, 1).view(batch, channels, height, width)


class MultiScaleStem(nn.Module):
    def __init__(self, in_planes: int, channels: int):
        super().__init__()
        branch_ch = channels // 3
        rem = channels - branch_ch * 3
        groups_3 = max(1, min(8, branch_ch // 8))
        while branch_ch % groups_3 != 0:
            groups_3 -= 1
        groups_5 = max(1, min(8, branch_ch // 8))
        while branch_ch % groups_5 != 0:
            groups_5 -= 1
        out_ch = branch_ch + rem
        groups_7 = max(1, min(8, out_ch // 8))
        while out_ch % groups_7 != 0:
            groups_7 -= 1
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_planes, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups_3, branch_ch),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_planes, branch_ch, 5, padding=2, bias=False),
            nn.GroupNorm(groups_5, branch_ch),
            nn.ReLU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_planes, out_ch, 7, padding=3, bias=False),
            nn.GroupNorm(groups_7, out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch3(x), self.branch5(x), self.branch7(x)], dim=1)


class ModelV8(nn.Module):
    def __init__(
        self,
        board_size: int = 15,
        channels: int = 300,
        num_res_blocks: int = 20,
        dropout: float = 0.1,
        use_coords: bool = True,
        use_checkpoint: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.board_size = board_size
        self.use_coords = use_coords
        self.use_checkpoint = use_checkpoint
        in_planes = 3 + (2 if use_coords else 0)
        groups = max(1, min(32, channels // 8))
        while channels % groups != 0:
            groups -= 1
        self.stem = MultiScaleStem(in_planes, channels)
        if num_res_blocks > 1:
            dpr = [drop_path_rate * i / (num_res_blocks - 1) for i in range(num_res_blocks)]
        else:
            dpr = [0.0]
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(PreNormResBlock(channels, groups=groups, dropout=dropout, drop_path=dpr[i]))
            if (i + 1) % 5 == 0:
                self.res_blocks.append(GlobalContextBlock(channels, reduction=4))
        self.transformer = TransformerBlock(channels, num_heads=8, mlp_ratio=2, dropout=dropout)
        self.final_norm = nn.GroupNorm(groups, channels)
        policy_ch = 96
        policy_groups = max(1, min(8, policy_ch // 8))
        self.policy_local = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 3, padding=1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
        )
        self.policy_global = nn.Sequential(
            nn.Conv2d(channels, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
        )
        self.policy_combine = nn.Sequential(
            nn.Conv2d(policy_ch * 2, policy_ch, 1, bias=False),
            nn.GroupNorm(policy_groups, policy_ch),
            nn.ReLU(),
            nn.Conv2d(policy_ch, 1, 1, bias=True),
        )
        value_ch = 96
        value_groups = max(1, min(8, value_ch // 8))
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_ch, 1, bias=False),
            nn.GroupNorm(value_groups, value_ch),
            nn.ReLU(),
        )
        self.value_attn = nn.Conv2d(value_ch, 1, 1, bias=True)
        value_hidden = 512
        self.value_fc = nn.Sequential(
            nn.Linear(value_ch, value_hidden),
            nn.LayerNorm(value_hidden),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(value_hidden, value_hidden // 2),
            nn.LayerNorm(value_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(value_hidden // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for block in self.res_blocks:
            if self.use_checkpoint and x.requires_grad and isinstance(block, PreNormResBlock):
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.transformer, x, use_reentrant=False)
        else:
            x = self.transformer(x)
        x = torch.relu(self.final_norm(x))
        p_local = self.policy_local(x)
        p_global = self.policy_global(x)
        policy = self.policy_combine(torch.cat([p_local, p_global], dim=1))
        policy = policy.view(policy.size(0), -1)
        v = self.value_conv(x)
        batch, channels, height, width = v.shape
        v_flat = v.view(batch, channels, height * width)
        attn = self.value_attn(v)
        attn = attn.view(batch, 1, height * width)
        attn = torch.softmax(attn, dim=-1)
        v_pooled = (v_flat * attn).sum(dim=-1)
        value = self.value_fc(v_pooled)
        value = torch.tanh(value).squeeze(-1)
        return value, policy


def get_distant_blocks(board: np.ndarray, player: int, win_length: int = 5) -> Set[Tuple[int, int]]:
    """For rule_type=0, find blocks that create a 2-ends-blocked scenario for opponent's 4-in-a-row."""
    distant_blocks = set()
    board_rows, board_cols = board.shape
    opponent = -player
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    for row in range(board_rows):
        for col in range(board_cols):
            if board[row, col] != opponent:
                continue
            for dr, dc in directions:
                count = 1
                for i in range(1, win_length):
                    nr, nc = row + dr * i, col + dc * i
                    if 0 <= nr < board_rows and 0 <= nc < board_cols and board[nr, nc] == opponent:
                        count += 1
                    else:
                        break
                
                if count == win_length - 1: # Found 4 stones in a row
                    end1_r, end1_c = row - dr, col - dc
                    end2_r, end2_c = row + dr * count, col + dc * count
                    
                    val1 = board[end1_r, end1_c] if (0 <= end1_r < board_rows and 0 <= end1_c < board_cols) else player
                    val2 = board[end2_r, end2_c] if (0 <= end2_r < board_rows and 0 <= end2_c < board_cols) else player
                    
                    # If end1 is blocked, and end2 is empty
                    if val1 == player and val2 == 0:
                        next_r, next_c = end2_r + dr, end2_c + dc
                        if 0 <= next_r < board_rows and 0 <= next_c < board_cols and board[next_r, next_c] == 0:
                            distant_blocks.add((next_r, next_c))
                            
                    # If end2 is blocked, and end1 is empty
                    if val2 == player and val1 == 0:
                        prev_r, prev_c = end1_r - dr, end1_c - dc
                        if 0 <= prev_r < board_rows and 0 <= prev_c < board_cols and board[prev_r, prev_c] == 0:
                            distant_blocks.add((prev_r, prev_c))
    return distant_blocks


def find_tactical_moves(
    board: np.ndarray, player: int, win_length: int = 5, rule_type: int = 0
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    board_rows, board_cols = board.shape
    winning_moves: Set[Tuple[int, int]] = set()
    blocking_moves: Set[Tuple[int, int]] = set()
    empty_positions = list(zip(*np.where(board == 0)))
    for row, col in empty_positions:
        board[row, col] = player
        if check_win_adaptive(board, row, col, player, win_length, rule_type):
            winning_moves.add((row, col))
        board[row, col] = 0
        board[row, col] = -player
        if check_win_adaptive(board, row, col, -player, win_length, rule_type):
            blocking_moves.add((row, col))
        board[row, col] = 0
        
    if rule_type == 0:
        blocking_moves.update(get_distant_blocks(board, player, win_length))
        
    return winning_moves, blocking_moves


def find_open4_threats(board: np.ndarray, player: int, win_length: int = 5) -> Set[Tuple[int, int]]:
    board_rows, board_cols = board.shape
    block_positions: Set[Tuple[int, int]] = set()
    opponent = -player
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for row in range(board_rows):
        for col in range(board_cols):
            if board[row, col] != opponent:
                continue
            for dr, dc in directions:
                count = 1
                for i in range(1, win_length):
                    nr, nc = row + dr * i, col + dc * i
                    if 0 <= nr < board_rows and 0 <= nc < board_cols and board[nr, nc] == opponent:
                        count += 1
                    else:
                        break
                if count == win_length - 1:
                    end1_r, end1_c = row - dr, col - dc
                    end2_r, end2_c = row + dr * count, col + dc * count
                    end1_open = (0 <= end1_r < board_rows and 0 <= end1_c < board_cols and board[end1_r, end1_c] == 0)
                    end2_open = (0 <= end2_r < board_rows and 0 <= end2_c < board_cols and board[end2_r, end2_c] == 0)
                    if end1_open:
                        block_positions.add((end1_r, end1_c))
                    if end2_open:
                        block_positions.add((end2_r, end2_c))
    return block_positions


def count_line_pattern(
    board: np.ndarray, row: int, col: int, player: int, dr: int, dc: int
) -> Tuple[int, int, int]:
    """Count stones, open ends, and gaps in a line pattern.
    Returns (stone_count, open_ends, gap_count)"""
    board_rows, board_cols = board.shape
    stones = 1
    open_ends = 0
    gaps = 0
    
    # Forward direction
    for i in range(1, 6):
        nr, nc = row + dr * i, col + dc * i
        if not (0 <= nr < board_rows and 0 <= nc < board_cols):
            break
        if board[nr, nc] == player:
            stones += 1
        elif board[nr, nc] == 0:
            # Check if there's a stone after the gap
            nr2, nc2 = row + dr * (i + 1), col + dc * (i + 1)
            if (0 <= nr2 < board_rows and 0 <= nc2 < board_cols and 
                board[nr2, nc2] == player and gaps == 0):
                gaps += 1
            else:
                open_ends += 1
                break
        else:
            break
    
    # Backward direction
    for i in range(1, 6):
        nr, nc = row - dr * i, col - dc * i
        if not (0 <= nr < board_rows and 0 <= nc < board_cols):
            break
        if board[nr, nc] == player:
            stones += 1
        elif board[nr, nc] == 0:
            nr2, nc2 = row - dr * (i + 1), col - dc * (i + 1)
            if (0 <= nr2 < board_rows and 0 <= nc2 < board_cols and 
                board[nr2, nc2] == player and gaps == 0):
                gaps += 1
            else:
                open_ends += 1
                break
        else:
            break
    
    return stones, open_ends, gaps


def evaluate_move_threats(board: np.ndarray, row: int, col: int, player: int) -> Tuple[int, int, int, int]:
    """Evaluate threats created by placing a stone at (row, col).
    Returns (win_threats, open4_count, open3_count, semi_open3_count)"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    win_threats = 0
    open4_count = 0
    open3_count = 0
    semi_open3_count = 0
    
    board[row, col] = player
    
    for dr, dc in directions:
        stones, open_ends, gaps = count_line_pattern(board, row, col, player, dr, dc)
        
        if stones >= 5:
            win_threats += 1
        elif stones == 4:
            if open_ends >= 2 or (open_ends >= 1 and gaps >= 1):
                open4_count += 1
            elif open_ends >= 1:
                semi_open3_count += 1  # Actually semi-open 4
        elif stones == 3:
            if open_ends >= 2:
                open3_count += 1
            elif open_ends >= 1:
                semi_open3_count += 1
    
    board[row, col] = 0
    return win_threats, open4_count, open3_count, semi_open3_count


def find_double_threats(
    board: np.ndarray, player: int, win_length: int = 5
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Find moves that create multiple threats simultaneously.
    Returns (double_open4, open4_open3, double_open3)"""
    board_rows, board_cols = board.shape
    double_open4: Set[Tuple[int, int]] = set()  # Creates 2+ open-4s (winning)
    open4_open3: Set[Tuple[int, int]] = set()   # Creates open-4 + open-3 (very strong)
    double_open3: Set[Tuple[int, int]] = set()  # Creates 2+ open-3s (strong)
    
    empty_positions = list(zip(*np.where(board == 0)))
    
    for row, col in empty_positions:
        # Check if position has neighbors (optimization)
        has_neighbor = False
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < board_rows and 0 <= nc < board_cols and board[nr, nc] != 0:
                    has_neighbor = True
                    break
            if has_neighbor:
                break
        
        if not has_neighbor:
            continue
        
        _, open4, open3, _ = evaluate_move_threats(board, row, col, player)
        
        if open4 >= 2:
            double_open4.add((row, col))
        elif open4 >= 1 and open3 >= 1:
            open4_open3.add((row, col))
        elif open3 >= 2:
            double_open3.add((row, col))
    
    return double_open4, open4_open3, double_open3


def find_forcing_moves(
    board: np.ndarray, player: int, win_length: int = 5
) -> Dict[str, Set[Tuple[int, int]]]:
    """Find all forcing moves categorized by priority."""
    winning, blocking = find_tactical_moves(board, player, win_length)
    open4_blocks = find_open4_threats(board, player, win_length)
    double_open4, open4_open3, double_open3 = find_double_threats(board, player, win_length)
    
    # Also check opponent's double threats to block
    opp_double_open4, opp_open4_open3, opp_double_open3 = find_double_threats(
        board, -player, win_length
    )
    
    return {
        "winning": winning,
        "blocking": blocking,
        "open4_blocks": open4_blocks,
        "double_open4": double_open4,
        "open4_open3": open4_open3,
        "double_open3": double_open3,
        "block_opp_double_open4": opp_double_open4,
        "block_opp_open4_open3": opp_open4_open3,
        "block_opp_double_open3": opp_double_open3,
    }


class MCTSNode:
    __slots__ = ("P", "N", "W", "virtual_loss", "children", "is_expanded")

    def __init__(self, prior: float = 0.0):
        self.P = float(prior)
        self.N = 0
        self.W = 0.0
        self.virtual_loss = 0
        self.children: Dict[Tuple[int, int], "MCTSNode"] = {}
        self.is_expanded = False

    def Q(self) -> float:
        return (self.W - self.virtual_loss) / max(1, self.N + self.virtual_loss)


class ProgressiveMCTS:
    def __init__(
        self,
        net: nn.Module,
        board_rows: int = 15,
        board_cols: int = 15,
        device: torch.device | str = "cpu",
        c_puct: float = 2.5,
        dirichlet_alpha: Optional[float] = None,
        noise_eps: float = 0.35,
        batch_size: int = 128,
        win_length: int = 5,
        progressive_widening: bool = True,
        use_coords: bool = False,
        rule_type: int = 0,
    ):
        self.net = net
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.device = device
        self.c_puct = c_puct
        self.noise_eps = noise_eps
        self.batch_size = batch_size
        self.win_length = win_length
        self.progressive_widening = progressive_widening
        self.rule_type = rule_type
        self.dir_alpha = 15.0 / min(board_rows, board_cols) if dirichlet_alpha is None else dirichlet_alpha
        self.root = MCTSNode()
        self.use_coords = use_coords
        self._state_channels = 4 + (2 if use_coords else 0)
        self._states_buffer = np.empty(
            (self.batch_size, self._state_channels, self.board_rows, self.board_cols), dtype=np.float32
        )

    def reset_root(self) -> None:
        self.root = MCTSNode()

    def move_root(self, move: Tuple[int, int]) -> bool:
        if self.root and move in self.root.children:
            new_root = self.root.children[move]
            new_root.virtual_loss = 0
            for child in new_root.children.values():
                child.virtual_loss = 0
            self.root = new_root
            self.root.is_expanded = True
            return True
        self.reset_root()
        return False

    def progressive_widen(self, priors: Dict[Tuple[int, int], float], num_visits: int) -> Dict[Tuple[int, int], float]:
        if not self.progressive_widening:
            return priors
        max_children = max(5, int(3 * math.sqrt(num_visits + 1)))
        max_children = min(max_children, len(priors))
        sorted_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)
        top_moves = dict(sorted_moves[:max_children])
        total = sum(top_moves.values())
        if total <= 0:
            return priors
        return {key: value / total for key, value in top_moves.items()}

    def add_dirichlet_noise(self, priors: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
        moves = list(priors.keys())
        if not moves:
            return priors
        noise = np.random.dirichlet([self.dir_alpha] * len(moves))
        new_priors: Dict[Tuple[int, int], float] = {}
        for i, move in enumerate(moves):
            new_priors[move] = (1 - self.noise_eps) * priors[move] + self.noise_eps * float(noise[i])
        return new_priors

    def expand_leaf_nodes_batch(self, leaf_list: List[dict], boost_tactical: bool = True) -> List[Tuple[dict, float, Dict]]:
        if not leaf_list:
            return []
        batch_size = len(leaf_list)
        if batch_size <= self.batch_size:
            states = self._states_buffer[:batch_size]
        else:
            states = np.empty((batch_size, self._state_channels, self.board_rows, self.board_cols), dtype=np.float32)
        for i, leaf in enumerate(leaf_list):
            states[i] = state_to_tensor(leaf["board"], leaf["player"], use_coords=self.use_coords, rule_type=self.rule_type)
        self.net.eval()
        x = torch.from_numpy(states)
        device_type = self.device.type if hasattr(self.device, "type") else str(self.device).split(":")[0]
        if device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        else:
            x = x.to(self.device)
        with torch.inference_mode():
            v_preds, p_logits = self.net(x)
            v_vals = v_preds.cpu().numpy().astype(np.float32)
            p_logits = p_logits.cpu().numpy().astype(np.float32)
        results: List[Tuple[dict, float, Dict]] = []
        for i, leaf in enumerate(leaf_list):
            board = leaf["board"]
            cur_player = leaf["player"]
            node = leaf["node"]
            logits = p_logits[i]
            mask = (board.reshape(-1) == 0).astype(np.float32)
            probs = np.exp(logits - np.max(logits)) * mask
            total = probs.sum()
            if total <= 1e-8:
                probs = mask
                total = probs.sum()
            probs = probs / (total + 1e-8)
            priors: Dict[Tuple[int, int], float] = {}
            for idx, prob in enumerate(probs):
                if mask[idx] > 0:
                    row, col = divmod(idx, self.board_cols)
                    priors[(row, col)] = float(prob)
            priors = filter_edge_moves(board, priors)
            
            if self.rule_type == 1:
                keys_to_remove = []
                for move in priors:
                    if is_useless_move(board, move[0], move[1], cur_player, self.win_length, self.rule_type):
                        keys_to_remove.append(move)
                if len(keys_to_remove) < len(priors):
                    for move in keys_to_remove:
                        del priors[move]
                    if keys_to_remove:
                        total_p = sum(priors.values())
                        if total_p > 0:
                            priors = {k: v / total_p for k, v in priors.items()}
            
            # Initialize tactical variables
            winning_moves: Set[Tuple[int, int]] = set()
            blocking_moves: Set[Tuple[int, int]] = set()
            
            if boost_tactical and priors:
                winning_moves, blocking_moves = find_tactical_moves(board.copy(), cur_player, self.win_length, self.rule_type)
                tactical_set = winning_moves | blocking_moves
                
                if tactical_set:
                    for move in tactical_set:
                        if move in priors:
                            priors[move] = 100.0
                
                total = sum(priors.values())
                if total > 0:
                    priors = {key: value / total for key, value in priors.items()}
            node.is_expanded = True
            node.children = {move: MCTSNode(prior=prior) for move, prior in priors.items()}
            
            # Adjust value based on tactical situation
            adjusted_value = float(np.squeeze(v_vals[i]))
            if winning_moves:
                adjusted_value = 0.95
            
            results.append((leaf, adjusted_value, priors))
        return results

    def backup_path(self, path: List[Tuple[MCTSNode, Tuple[int, int]]], leaf_value: float) -> None:
        value = leaf_value
        for node, move in reversed(path):
            child = node.children.get(move)
            if child is None:
                continue
            child.N += 1
            child.W += value
            child.virtual_loss -= 1
            value = -value

    def run_simulations(
        self,
        board: np.ndarray,
        player: int,
        num_sims: int = 200,
        add_noise: bool = True,
        temperature: float = 1.0,
    ) -> Dict[Tuple[int, int], int]:
        root = self.root
        if not root.is_expanded:
            leaf_batch = [{"board": board, "player": player, "node": root, "path": []}]
            batch_results = self.expand_leaf_nodes_batch(leaf_batch)
            if batch_results:
                _, _, priors = batch_results[0]
                if add_noise:
                    priors = self.add_dirichlet_noise(priors)
                root.is_expanded = True
                root.children = {move: MCTSNode(prior=prior) for move, prior in priors.items()}
        leaf_batch: List[dict] = []
        sims_done = 0
        while sims_done < num_sims:
            node = root
            cur_board = board.copy()
            cur_player = player
            path: List[Tuple[MCTSNode, Tuple[int, int]]] = []
            terminal = False
            while node.is_expanded and node.children:
                total_visits = sum(child.N for child in node.children.values())
                widened_priors = self.progressive_widen({mv: c.P for mv, c in node.children.items()}, total_visits)
                candidate_moves = set(widened_priors.keys())
                best_move = None
                best_score = -1e9
                sqrt_sum = math.sqrt(max(1, sum(node.children[m].N for m in candidate_moves)))
                for move in candidate_moves:
                    child = node.children[move]
                    u = self.c_puct * child.P * (sqrt_sum / (1 + child.N + child.virtual_loss))
                    q = child.Q()
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_move = move
                move = best_move
                if move is None:
                    break
                child = node.children[move]
                child.virtual_loss += 1
                path.append((node, move))
                row, col = move
                cur_board[row, col] = cur_player
                if check_win_adaptive(cur_board, row, col, cur_player, self.win_length, self.rule_type):
                    leaf_value = 1.0
                    self.backup_path(path, leaf_value)
                    terminal = True
                    break
                if np.all(cur_board != 0):
                    self.backup_path(path, 0.0)
                    terminal = True
                    break
                cur_player = -cur_player
                node = child
            if terminal:
                sims_done += 1
                continue
            leaf_batch.append({"board": cur_board, "player": cur_player, "node": node, "path": path})
            current_batch_size = min(self.batch_size, max(1, num_sims // 10))
            if len(leaf_batch) >= current_batch_size or sims_done + len(leaf_batch) >= num_sims:
                batch_results = self.expand_leaf_nodes_batch(leaf_batch)
                for leaf, v_pred, _ in batch_results:
                    leaf_value = -float(v_pred)
                    self.backup_path(leaf["path"], leaf_value)
                sims_done += len(leaf_batch)
                import time
                time.sleep(0.005) # Yield GIL
                leaf_batch = []
        if root.children:
            return {move: child.N for move, child in root.children.items()}
        return {}


def load_model(model_path: str, board_size: int, device: torch.device) -> nn.Module:
    if str(model_path).endswith(".onnx"):
        print(f"Loading ONNX model: {model_path}")
        return ONNXNetWrapper(str(model_path)).to(device)
        
    model = ModelV8(
        board_size=board_size,
        channels=320,
        num_res_blocks=20,
        dropout=0.1,
        use_coords=True,
    ).to(device)
    checkpoint_data = torch.load(model_path, map_location=device, weights_only=False)
    if "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
    elif "model_state_dict" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model.load_state_dict(checkpoint_data)
    model.eval()
    return model


@dataclass
class GameSession:
    board_rows: int = 15
    board_cols: int = 15
    human_player: int = -1
    sims: int = 400
    model: nn.Module = None
    device: torch.device = None
    rule_type: int = 0
    win_length: int = 5
    
    board: np.ndarray = field(init=False)
    current_player: int = field(init=False, default=1)
    winner: Optional[int] = field(init=False, default=None)
    last_move: Optional[Tuple[int, int]] = field(init=False, default=None)
    mcts: ProgressiveMCTS = field(init=False)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.last_move = None
        self.mcts = ProgressiveMCTS(
            net=self.model,
            board_rows=self.board_rows,
            board_cols=self.board_cols,
            device=self.device,
            c_puct=3.5,
            batch_size=128,
            win_length=self.win_length,
            progressive_widening=True,
            use_coords=True,
            rule_type=self.rule_type,
        )
        self.mcts.reset_root()
        self.lock = threading.Lock()

    def apply_move(self, row: int, col: int) -> None:
        if self.winner is not None:
            raise ValueError("Game already finished.")
        if self.board[row, col] != 0:
            raise ValueError("Cell is occupied.")
        self.board[row, col] = self.current_player
        self.last_move = (row, col)
        self.mcts.move_root((row, col))
        if check_win_adaptive(self.board, row, col, self.current_player, self.win_length, self.rule_type):
            self.winner = self.current_player
        elif np.all(self.board != 0):
            self.winner = 0
        else:
            self.current_player *= -1

    @property
    def current_mcts_probs(self) -> List[dict]:
        if not hasattr(self, 'mcts') or not self.mcts.root or not self.mcts.root.children:
            return []
        counts = {move: child.N for move, child in self.mcts.root.children.items()}
        total = sum(counts.values())
        if total == 0:
            return []
        sorted_moves = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        top_moves = []
        for m, count in sorted_moves[:5]:
            prob = count / total
            if prob > 0.01:
                top_moves.append({"row": m[0], "col": m[1], "prob": float(prob)})
        return top_moves

    def ai_move(self) -> Tuple[Optional[Tuple[int, int]], List[dict], float]:
        if self.winner is not None:
            return None, [], 0.0
            
        num_pieces = np.count_nonzero(self.board)
            
        counts = self.mcts.run_simulations(
            self.board.copy(),
            self.current_player,
            num_sims=self.sims,
            add_noise=False, # Disable noise for actual gameplay
        )
        if not counts:
            return None, [], 0.0
            
        move = max(counts, key=counts.get)
        
        # Calculate evaluation score from Player 1's perspective
        # Use the best child's Q value, as it represents the expected outcome for the current player
        best_child = self.mcts.root.children[move]
        eval_score = best_child.Q()
        if self.current_player == -1:
            eval_score = -eval_score
            
        total_visits = sum(counts.values())
        top_moves = []
        if total_visits > 0:
            sorted_moves = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for m, count in sorted_moves[:5]:
                prob = count / total_visits
                if prob > 0.01:
                    top_moves.append({"row": m[0], "col": m[1], "prob": float(prob)})
            
        self.apply_move(*move)
        return move, top_moves, float(eval_score)
