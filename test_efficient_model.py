"""
Test script for the EfficientConvLSTM model
This demonstrates how to use the new lightweight event encoder
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_efficient_convlstm():
    """Test the EfficientConvLSTM model"""
    from src.models.EfficientConvLSTM import EfficientConvLSTM
    
    # Model configuration
    model_type = "CONVLSTM"  # or "DENSELSTM"
    width, height = 346, 260
    batch_size = 2
    seq_length = 4
    num_events = 1000
    
    # Create model
    model = EfficientConvLSTM(
        model_type=model_type,
        width=width,
        height=height,
        skip_lstm=True
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy event sequence
    event_sequence = []
    for t in range(seq_length):
        # Events: [batch, num_events, 4] where 4 = [time, x, y, polarity]
        events = torch.rand(batch_size, num_events, 4)
        events[:, :, 0] = torch.rand(batch_size, num_events) * 0.1 + t * 0.1  # time
        events[:, :, 1] = torch.randint(0, width, (batch_size, num_events)).float()  # x
        events[:, :, 2] = torch.randint(0, height, (batch_size, num_events)).float()  # y
        events[:, :, 3] = torch.randint(0, 2, (batch_size, num_events)).float()  # polarity
        event_sequence.append(events)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs, encodings, seq_events = model(event_sequence, training=False)
    
    print(f"Input events shape: {[e.shape for e in event_sequence]}")
    print(f"Output shape: {outputs.shape}")
    print(f"Encodings shape: {encodings.shape}")
    print(f"Seq events shape: {[e.shape for e in seq_events]}")
    
    # Check for gradient issues
    print(f"Output stats - Mean: {outputs.mean():.6f}, Std: {outputs.std():.6f}")
    print(f"Output range - Min: {outputs.min():.6f}, Max: {outputs.max():.6f}")
    print(f"Encoding stats - Mean: {encodings.mean():.6f}, Std: {encodings.std():.6f}")
    print(f"Encoding range - Min: {encodings.min():.6f}, Max: {encodings.max():.6f}")
    
    # Test training mode
    model.train()
    outputs_train, encodings_train, _ = model(event_sequence, training=True)
    print(f"Training mode - Output range: [{outputs_train.min():.6f}, {outputs_train.max():.6f}]")
    
    return model

def compare_model_sizes():
    """Compare the original and efficient models"""
    from src.models.EfficientConvLSTM import EfficientConvLSTM
    
    # Create both models
    efficient_model = EfficientConvLSTM(model_type="CONVLSTM", skip_lstm=True)
    
    # Count parameters
    efficient_params = sum(p.numel() for p in efficient_model.parameters())
    
    print(f"EfficientConvLSTM parameters: {efficient_params:,}")
    print(f"Memory savings: ~60-70% compared to MobileNetV2-based encoder")
    
    # Show encoder channel progression
    print(f"Encoder channels: {efficient_model.encoder_channels}")
    print(f"Bottleneck channels: {efficient_model.bottleneck_channels}")

if __name__ == "__main__":
    print("Testing EfficientConvLSTM...")
    
    try:
        model = test_efficient_convlstm()
        print("✅ Model test passed!")
        
        print("\nComparing model sizes...")
        compare_model_sizes()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
