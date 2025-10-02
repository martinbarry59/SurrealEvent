import torch
import torch.nn as nn
from src.models.EventSegFast import EventSegFast

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_memory_usage(model, batch_size=10, sequence_length=25, height=260, width=346, voxel_channels=5):
    """Estimate memory usage for forward pass"""
    model.eval()
    
    # Calculate activation memory for one timestep
    dummy_input = torch.randn(batch_size, 4, 1000)  # events
    
    # Simulate voxel conversion
    voxel_size = batch_size * voxel_channels * height * width * 4  # float32
    
    # Calculate intermediate feature sizes - NOW ALL AT 4x DOWNSAMPLED RESOLUTION
    base_ch = 24  # Updated base channels
    ds_height, ds_width = height // 4, width // 4  # Downsampled dimensions
    
    # VoxelEmbedding output: [B, 24, H/4, W/4] - MUCH SMALLER!
    embed_size = batch_size * base_ch * ds_height * ds_width * 4
    
    # Encoder features - all at downsampled resolution
    f1_size = batch_size * base_ch * ds_height * ds_width * 4  # [B, 24, H/4, W/4]
    f2_size = batch_size * (base_ch*2) * (ds_height//2) * (ds_width//2) * 4  # [B, 48, H/8, W/8]
    f3_size = batch_size * (base_ch*4) * (ds_height//4) * (ds_width//4) * 4  # [B, 96, H/16, W/16]
    
    # CrossScaleFusion: 48 channels per scale -> 144 total -> 64 output at H/4, W/4
    fusion_size = batch_size * 64 * ds_height * ds_width * 4
    
    # Final output
    output_size = batch_size * 1 * height * width * 4
    
    # Total per timestep (bytes)
    per_timestep = voxel_size + embed_size + f1_size + f2_size + f3_size + fusion_size + output_size
    
    # For sequence
    total_sequence = per_timestep * sequence_length
    
    # Convert to GB
    per_timestep_gb = per_timestep / (1024**3)
    total_sequence_gb = total_sequence / (1024**3)
    
    print(f"Memory per timestep: {per_timestep_gb:.3f} GB")
    print(f"Memory for {sequence_length} timesteps: {total_sequence_gb:.3f} GB")
    print(f"Voxel size per step: {voxel_size / (1024**3):.3f} GB")
    print(f"Feature sizes per step (ALL AT H/4 x W/4 = {ds_height}x{ds_width}):")
    print(f"  Embed: {embed_size / (1024**3):.3f} GB")
    print(f"  F1: {f1_size / (1024**3):.3f} GB") 
    print(f"  F2: {f2_size / (1024**3):.3f} GB")
    print(f"  F3: {f3_size / (1024**3):.3f} GB")
    print(f"  Fusion: {fusion_size / (1024**3):.3f} GB")
    print(f"  Final output (upsampled): {output_size / (1024**3):.3f} GB")
    
    return per_timestep_gb, total_sequence_gb

def analyze_model_components(model):
    """Analyze each component's parameter count"""
    print("Component analysis:")
    
    total, trainable = count_parameters(model.embed)
    print(f"VoxelEmbedding: {total:,} params")
    
    total, trainable = count_parameters(model.temporal_agg)
    print(f"TemporalAggregator: {total:,} params")
    
    total, trainable = count_parameters(model.temporal_mem)
    print(f"TemporalMemory: {total:,} params")
    
    total, trainable = count_parameters(model.enc1)
    print(f"Encoder1: {total:,} params")
    
    total, trainable = count_parameters(model.enc2)
    print(f"Encoder2: {total:,} params")
    
    total, trainable = count_parameters(model.enc3)
    print(f"Encoder3: {total:,} params")
    
    total, trainable = count_parameters(model.fusion)
    print(f"CrossScaleFusion: {total:,} params")
    
    total, trainable = count_parameters(model.head)
    print(f"SegmentationHead: {total:,} params")

if __name__ == "__main__":
    # Create model with reduced config
    model = EventSegFast(voxel_channels=5, height=260, width=346, base_ch=24)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024**2):.2f}")
    print()
    
    analyze_model_components(model)
    print()
    
    estimate_memory_usage(model, batch_size=10, sequence_length=5)
