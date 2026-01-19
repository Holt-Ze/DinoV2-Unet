import timm
print("timm version:", timm.__version__)
try:
    models = timm.list_models('*vit_base*')
    for m in models:
        print(m)
except Exception as e:
    print(f"Error listing models: {e}")
