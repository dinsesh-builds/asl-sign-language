import pickle, json, numpy as np

print("=" * 50)
print("  ASL Model Exporter")
print("=" * 50)

# ── Load model ──────────────────────────────────────
print("\n[1/4] Loading model.p ...")
d     = pickle.load(open('model.p', 'rb'))
model = d['model']
n     = d.get('n_features', 63)
print(f"      Pipeline steps : {[s for s,_ in model.steps]}")
print(f"      Feature count  : {n}")

# ── Extract CORRECT label order from the SVM ────────
print("\n[2/4] Reading class labels from trained SVM ...")
svm        = model.named_steps['svm']
raw_labels = list(svm.classes_)          # exact order SVM uses internally
print(f"      Raw SVM classes (first 10): {raw_labels[:10]}")
print(f"      Total classes: {len(raw_labels)}")

# Build sign map: folder number → letter/digit
sign_map = {}
for i in range(26): sign_map[i]    = chr(65 + i)   # 0→A, 1→B, ..., 25→Z
for i in range(10): sign_map[26+i] = str(i)         # 26→0, 27→1, ..., 35→9

# Convert raw label (string like '0','10','25') → sign character
# This preserves the EXACT order the ONNX model outputs probabilities in
labels_in_order = []
for raw in raw_labels:
    idx  = int(raw)
    char = sign_map.get(idx, '?')
    labels_in_order.append(char)

print(f"      Labels in ONNX order (first 10): {labels_in_order[:10]}")
print(f"      Full label list: {labels_in_order}")

# ── Save labels.json ─────────────────────────────────
print("\n[3/4] Saving labels.json ...")
with open('labels.json', 'w') as f:
    json.dump(labels_in_order, f)
print("      labels.json saved ✓")

# ── Export ONNX ───────────────────────────────────────
print("\n[4/4] Exporting model.onnx ...")
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, n]))]
    onnx_model   = convert_sklearn(
        model,
        initial_types=initial_type,
        options={type(svm): {'zipmap': False}}  # returns raw array not dict
    )

    with open('model.onnx', 'wb') as f:
        onnx_bytes = onnx_model.SerializeToString()  # type: ignore[union-attr]
        f.write(onnx_bytes)

    print("      model.onnx saved ✓")
    print()
    print("=" * 50)
    print("  ✓  Export complete!")
    print("=" * 50)
    print()
    print("  Files generated:")
    print("    model.onnx   ← ONNX model for browser")
    print("    labels.json  ← correct sign mapping")
    print()
    print("  Both files must be in the same folder as asl_live.html")
    print()
    print("  Open Edge → http://localhost:5500/asl_live.html")
    print("  Load model.onnx when prompted")
    print()

except ImportError:
    print("\n  ✗  skl2onnx not installed. Run:")
    print("     pip install skl2onnx onnx\n")
except Exception as e:
    print(f"\n  ✗  Export failed: {e}\n")