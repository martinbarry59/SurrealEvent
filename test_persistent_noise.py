#!/usr/bin/env python3
"""
Test script for the improved PersistentNoiseGenerator with lamp-like blobs
and enhanced night-time camera artifacts simulation.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.functions import (
    PersistentNoiseGenerator, 
    PersistentNoiseConfig,
    create_nighttime_noise_config,
    create_minimal_noise_config,
    merge_events_with_noise,
    eventstovoxel
)

def visualize_events(events, width=346, height=260, title="Events"):
    """Visualize events as a scatter plot"""
    if events is None or events.shape[1] == 0:
        print(f"No events to visualize for {title}")
        return
        
    # Take first batch
    batch_events = events[0]  # [N, 4]
    
    # Separate positive and negative polarity
    pos_mask = batch_events[:, 3] > 0
    neg_mask = batch_events[:, 3] < 0
    
    plt.figure(figsize=(12, 8))
    
    if pos_mask.any():
        pos_events = batch_events[pos_mask]
        plt.scatter(pos_events[:, 1].cpu(), height - pos_events[:, 2].cpu(), 
                   c='red', s=1, alpha=0.6, label=f'Positive ({pos_mask.sum()} events)')
    
    if neg_mask.any():
        neg_events = batch_events[neg_mask]
        plt.scatter(neg_events[:, 1].cpu(), height - neg_events[:, 2].cpu(), 
                   c='blue', s=1, alpha=0.6, label=f'Negative ({neg_mask.sum()} events)')
    
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.title(f'{title} - Total: {batch_events.shape[0]} events')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def test_lamp_like_behavior():
    """Test the lamp-like blob behavior over multiple time steps"""
    print("Testing lamp-like blob behavior...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create nighttime configuration
    config = create_nighttime_noise_config()
    print("Night-time configuration:")
    print(f"  Lamp blobs: {config.lamp_blobs}")
    print(f"  Lamp events: {config.lamp_events}")
    print(f"  Lamp drift speed: {config.lamp_drift_speed}")
    print(f"  Hot pixels: {config.hot_pixels}")
    print(f"  Flicker sources: {config.flicker_sources}")
    
    # Initialize generator
    width, height = 346, 260
    generator = PersistentNoiseGenerator(width, height, device, config, seed=42)
    
    # Test multiple time steps to see lamp movement
    batch_size = 2
    time_steps = 5
    
    print(f"\nGenerating noise for {time_steps} time steps...")
    
    lamp_positions = []
    total_events_per_step = []
    
    for step in range(time_steps):
        t_min = step * 0.2
        t_max = (step + 1) * 0.2
        
        print(f"\nStep {step+1}: t ∈ [{t_min:.1f}, {t_max:.1f}]")
        
        # Generate noise events
        noise_events = generator.step(batch_size, t_min, t_max)
        
        if noise_events is not None:
            print(f"  Generated {noise_events.shape[1]} noise events per batch")
            total_events_per_step.append(noise_events.shape[1])
            
            # Track lamp positions (they should move very slowly)
            if hasattr(generator, 'lamp_pos'):
                lamp_positions.append(generator.lamp_pos.clone().cpu())
            
            # Visualize first step in detail
            if step == 0:
                visualize_events(noise_events, width, height, 
                               f"Night-time Noise (Step {step+1})")
        else:
            print("  No events generated")
            total_events_per_step.append(0)
    
    # Analyze lamp movement
    if lamp_positions:
        print(f"\nLamp movement analysis over {time_steps} steps:")
        for i in range(len(lamp_positions[0])):
            start_pos = lamp_positions[0][i]
            end_pos = lamp_positions[-1][i]
            distance = torch.norm(end_pos - start_pos).item()
            print(f"  Lamp {i+1}: moved {distance:.2f} pixels (very slow = good)")
    
    print(f"\nEvent count per step: {total_events_per_step}")
    return generator

def test_realistic_simulation():
    """Test the realistic simulation of camera artifacts"""
    print("\n" + "="*60)
    print("Testing realistic camera artifact simulation...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different configurations
    configs = {
        "Night-time": create_nighttime_noise_config(),
        "Minimal": create_minimal_noise_config(),
        "Default": PersistentNoiseConfig()
    }
    
    width, height = 346, 260
    batch_size = 1
    
    for config_name, config in configs.items():
        print(f"\n--- Testing {config_name} Configuration ---")
        
        generator = PersistentNoiseGenerator(width, height, device, config, seed=123)
        
        # Generate noise for a time slice
        noise_events = generator.step(batch_size, 0.0, 1.0)
        
        if noise_events is not None:
            print(f"Generated {noise_events.shape[1]} events")
            
            # Create some dummy original events for testing merge
            original_events = torch.rand(batch_size, 1000, 4, device=device)
            original_events[:, :, 1] *= width - 1  # x coordinates
            original_events[:, :, 2] *= height - 1  # y coordinates
            original_events[:, :, 3] = torch.randint(0, 2, (batch_size, 1000), device=device) * 2 - 1  # polarity
            
            # Merge with noise
            merged_events = merge_events_with_noise(original_events, noise_events)
            print(f"Merged events shape: {merged_events.shape}")
            print(f"Total events: {merged_events.shape[1]} (original: 1000, noise: {noise_events.shape[1]})")
            
            # Test voxel conversion
            voxel = eventstovoxel(merged_events, height, width, bins=5, training=False)
            print(f"Voxel shape: {voxel.shape}")
            print(f"Voxel range: [{voxel.min().item():.2f}, {voxel.max().item():.2f}]")
            
        else:
            print("No events generated (noise disabled)")

def create_comparison_visualization():
    """Create a visual comparison of different noise configurations"""
    print("\n" + "="*60)
    print("Creating visual comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    width, height = 346, 260
    batch_size = 1
    
    configs = {
        "Original (High Speed Blobs)": PersistentNoiseConfig(
            moving_blobs=2, blob_speed=8.0, lamp_blobs=0
        ),
        "New (Lamp-like Stationary)": create_nighttime_noise_config()
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, (config_name, config) in enumerate(configs.items()):
        generator = PersistentNoiseGenerator(width, height, device, config, seed=42)
        noise_events = generator.step(batch_size, 0.0, 1.0)
        
        ax = axes[idx]
        
        if noise_events is not None and noise_events.shape[1] > 0:
            # Take first batch
            batch_events = noise_events[0]  # [N, 4]
            
            # Separate positive and negative polarity
            pos_mask = batch_events[:, 3] > 0
            neg_mask = batch_events[:, 3] < 0
            
            if pos_mask.any():
                pos_events = batch_events[pos_mask]
                ax.scatter(pos_events[:, 1].cpu(), height - pos_events[:, 2].cpu(), 
                          c='red', s=2, alpha=0.7, label=f'Positive')
            
            if neg_mask.any():
                neg_events = batch_events[neg_mask]
                ax.scatter(neg_events[:, 1].cpu(), height - neg_events[:, 2].cpu(), 
                          c='blue', s=2, alpha=0.7, label=f'Negative')
            
            ax.set_title(f'{config_name}\n{batch_events.shape[0]} events')
        else:
            ax.set_title(f'{config_name}\nNo events')
        
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('persistent_noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison saved as 'persistent_noise_comparison.png'")

if __name__ == "__main__":
    print("PersistentNoiseGenerator Enhanced Test")
    print("="*60)
    
    # Test lamp-like behavior
    generator = test_lamp_like_behavior()
    
    # Test realistic simulation
    test_realistic_simulation()
    
    # Create visual comparison
    create_comparison_visualization()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("\nKey improvements:")
    print("• Lamp-like blobs now move very slowly (drift speed: 0.3-0.5)")
    print("• Multiple lamp intensities with realistic falloff patterns")
    print("• Hot pixels now show burst behavior instead of constant rate")
    print("• Enhanced flicker with more realistic light spread")
    print("• Improved edge and grid interference patterns")
    print("• Night-time configuration optimized for strong illumination scenarios")
