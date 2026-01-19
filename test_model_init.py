from seg.models import DinoV2UNet
import torch

try:
    print("Attempting to initialize DinoV2UNet with pretrained_type='imagenet_supervised'...")
    model = DinoV2UNet(backbone='vit_base_patch14_dinov2', pretrained_type='imagenet_supervised')
    print("Model initialized successfully!")
    print(f"Encoder Patch Size: {model.encoder.patch_size}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 3, 448, 448) # Standard size
    print(f"Testing forward pass with input shape {dummy_input.shape}...")
    output = model(dummy_input)
    print(f"Forward pass successful. Output shape: {output.shape}")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
