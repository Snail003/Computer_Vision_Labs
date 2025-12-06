import os
import random
import numpy as np
from tensorflow import keras

NP_DIR = os.path.join("dataset", "np_data")
EMB_MODEL_PATH = "embedding_model.keras"

NUM_SAME_PAIRS = 500
NUM_DIFF_PAIRS = 500
RANDOM_SEED = 42

FP_WEIGHT = 6.0
FN_WEIGHT = 1.0


def main():
    rng = random.Random(RANDOM_SEED)

    x_train = np.load(os.path.join(NP_DIR, "img_train.npy")).astype("float32") / 255.0
    y_train = np.load(os.path.join(NP_DIR, "label_train.npy")).reshape(-1)

    x_real = np.load(os.path.join(NP_DIR, "img_real.npy")).astype("float32") / 255.0
    y_real = np.load(os.path.join(NP_DIR, "label_real.npy")).reshape(-1)

    if x_train.ndim == 3:
        x_train = x_train[..., np.newaxis]
    if x_real.ndim == 3:
        x_real = x_real[..., np.newaxis]

    print("x_train:", x_train.shape, "y_train:", y_train.shape)
    print("x_real:", x_real.shape, "y_real:", y_real.shape)

    n_real = x_real.shape[0]

    emb_model = keras.models.load_model(EMB_MODEL_PATH)
    print("Embedding model loaded.")

    emb_real = emb_model.predict(x_real, batch_size=64, verbose=1)

    train_by_label = {}
    for idx, lbl in enumerate(y_train):
        train_by_label.setdefault(int(lbl), []).append(idx)

    same_pairs = []
    diff_pairs = []

    all_real_indices = list(range(n_real))

    while len(same_pairs) < NUM_SAME_PAIRS:
        ri = rng.choice(all_real_indices)
        lbl = int(y_real[ri])
        candidates = train_by_label.get(lbl, [])
        if not candidates:
            continue
        tj = rng.choice(candidates)
        same_pairs.append((ri, tj, 1))

    while len(diff_pairs) < NUM_DIFF_PAIRS:
        ri = rng.choice(all_real_indices)
        lbl_r = int(y_real[ri])
        diff_candidates = [idx for idx, lbl_t in enumerate(y_train) if int(lbl_t) != lbl_r]
        if not diff_candidates:
            continue
        tj = rng.choice(diff_candidates)
        diff_pairs.append((ri, tj, 0))

    all_pairs = same_pairs + diff_pairs
    rng.shuffle(all_pairs)

    dists_list = []
    labels_list = []

    for ri, tj, pair_lbl in all_pairs:
        real_emb = emb_real[ri]
        train_img = x_train[tj][np.newaxis]
        train_emb = emb_model.predict(train_img, batch_size=1, verbose=0)[0]

        dist = float(np.linalg.norm(train_emb - real_emb))
        dists_list.append(dist)
        labels_list.append(pair_lbl)

    dists = np.array(dists_list)
    pair_labels = np.array(labels_list, dtype=np.int32)
    d_min, d_max = dists.min(), dists.max()
    thresholds = np.linspace(d_min, d_max, 100)

    best_thr = None
    best_acc = 0.0
    best_cost = None

    for thr in thresholds:
        pred_same = (dists < thr).astype(np.int32)

        tp = np.sum((pred_same == 1) & (pair_labels == 1))
        tn = np.sum((pred_same == 0) & (pair_labels == 0))
        fp = np.sum((pred_same == 1) & (pair_labels == 0))
        fn = np.sum((pred_same == 0) & (pair_labels == 1))

        cost = FP_WEIGHT * fp + FN_WEIGHT * fn
        acc = (tp + tn) / len(pair_labels)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_thr = thr
            best_acc = acc

    print(f"\nBest threshold (considering errors): {best_thr:.4f}")
    print(f"Minimal weeighted cost of errors: {best_cost:.4f}")
    print(f"Accuracy of this threshold: {best_acc:.4f}")

    pred_same = (dists < best_thr).astype(np.int32)

    tp = np.sum((pred_same == 1) & (pair_labels == 1))
    tn = np.sum((pred_same == 0) & (pair_labels == 0))
    fp = np.sum((pred_same == 1) & (pair_labels == 0))
    fn = np.sum((pred_same == 0) & (pair_labels == 1))

    print("\nConfusion matrix:")
    print(f"TP (same & predicted same):       {tp}")
    print(f"TN (different & predicted diff.): {tn}")
    print(f"FP (different, but predicted same): {fp}")
    print(f"FN (same, but predicted diff.):     {fn}")

if __name__ == "__main__":
    main()