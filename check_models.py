import os, h5py
paths = ["disease_model.h5", "skin_type_model.h5"]
for p in paths:
    ap = os.path.abspath(p)
    print("Checking:", p, "->", ap)
    print("exists:", os.path.exists(p))
    if os.path.exists(p):
        try:
            with h5py.File(p, 'r') as f:
                print("top keys:", list(f.keys()))
        except Exception as e:
            print("h5py open error:", type(e).__name__, e)
    print("---")
