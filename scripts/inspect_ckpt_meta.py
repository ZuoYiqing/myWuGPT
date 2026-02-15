import torch

for name in ["weights/sft_test.pt", "weights/sft_nonviolent.pt"]:
    ck = None
    # try safe weights-only load first to avoid requiring repo modules during unpickling
    try:
        ck = torch.load(name, map_location='cpu', weights_only=True)
    except TypeError:
        # older torch may not support weights_only
        try:
            ck = torch.load(name, map_location='cpu')
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            continue
    except Exception as e:
        # fallback: attempt normal load but catch module import issues
        try:
            ck = torch.load(name, map_location='cpu')
        except Exception as e2:
            print(f"Failed to load {name}: {e2}")
            continue
    print("FILE:", name)
    print(" step:", ck.get("step"))
    print(" keys:", list(ck.keys()))
    ta = ck.get("train_args")
    if ta:
        print(" train_args.data:", ta.get("data"), " max_steps:", ta.get("max_steps"))
    print()
