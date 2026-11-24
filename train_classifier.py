import pickle, numpy as np, time
from sklearn.svm            import SVC
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics        import accuracy_score, classification_report
from collections            import Counter

print("=" * 55)
print("  ASL Classifier Training  (Improved)")
print("=" * 55)

# ── Load dataset ─────────────────────────────────────────
print("\n[1/5] Loading data.pickle ...")
d      = pickle.load(open('data.pickle', 'rb'))
X_raw  = np.array(d['data'])
y_raw  = np.array(d['labels'])

size_counts = Counter(len(x) for x in d['data'])
n_features  = max(size_counts, key=lambda k: size_counts[k])
mask        = np.array([len(x) == n_features for x in d['data']])
X_raw       = X_raw[mask]
y_raw       = y_raw[mask]

print(f"      Samples   : {len(X_raw)}")
print(f"      Features  : {n_features}")
print(f"      Classes   : {len(np.unique(y_raw))}")

# Split before augmentation so near-duplicates cannot leak into the test set.
print("\n[2/5] Splitting data (80/20) ...")
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=42
)
print(f"      Train: {len(X_train_raw)}  |  Test: {len(X_test)}")

print("\n[3/5] Augmenting training data only (3x size) ...")
np.random.seed(42)
X_aug_list = [X_train_raw]
y_aug_list = [y_train_raw]
for noise_level in [0.008, 0.015]:
    noise = np.random.normal(0, noise_level, X_train_raw.shape)
    X_aug_list.append(np.clip(X_train_raw + noise, 0, None))
    y_aug_list.append(y_train_raw)

X_train = np.vstack(X_aug_list)
y_train = np.concatenate(y_aug_list)
print(f"      Augmented training samples: {len(X_train)}")

# ── Hyperparameter search ─────────────────────────────────
print("\n[4/5] Finding best SVM parameters (2-5 min) ...")
t0 = time.time()
param_grid = {
    'svm__C':     [1, 5, 10, 50, 100],
    'svm__gamma': ['scale', 'auto'],
}
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', probability=True, random_state=42))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
best   = gs.best_estimator_
bpar   = gs.best_params_
bscore = gs.best_score_
print(f"      Best params : {bpar}")
print(f"      Best CV acc : {bscore*100:.2f}%")
print(f"      Time taken  : {time.time()-t0:.1f}s")

# ── Evaluate ──────────────────────────────────────────────
print("\n[5/5] Evaluating on test set ...")
sign_map = {}
for i in range(26): sign_map[str(i)]    = chr(65+i)
for i in range(10): sign_map[str(26+i)] = str(i)

y_pred  = best.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
classes = sorted(np.unique(y_test), key=lambda x: int(x))
names   = [sign_map.get(c, c) for c in classes]

print(f"\n  ── Test Accuracy: {acc*100:.2f}% ──\n")
print(classification_report(y_test, y_pred, labels=classes, target_names=names))

if acc < 1.0:
    print("  Confused sign pairs:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    for i, tc in enumerate(classes):
        for j, pc in enumerate(classes):
            if i != j and cm[i,j] > 0:
                print(f"    {sign_map.get(tc,'?')} mistaken as {sign_map.get(pc,'?')}: {cm[i,j]} times")

# ── Save ───────────────────────────────────────────────────
print("\n  Saving model.p ...")
with open('model.p', 'wb') as f:
    pickle.dump({
        'model':      best,
        'n_features': n_features,
        'labels':     [str(c) for c in classes],
        'accuracy':   acc
    }, f)

print()
print("=" * 55)
print(f"  Done! Accuracy: {acc*100:.2f}%")
print("=" * 55)
if acc >= 0.99:
    print("\n  Excellent! Run inference_classifier.py")
elif acc >= 0.90:
    print("\n  Good. Recollect data for confused signs above.")
else:
    print("\n  Recollect ALL signs with more pose variation.")
print("\n  Then run: python export_model.py  (to update browser too)")