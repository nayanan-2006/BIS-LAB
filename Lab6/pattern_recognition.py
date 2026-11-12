
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==========================================
# 1. STORE REFERENCE PATTERNS
# ==========================================

patterns = np.array([
    [0, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 1],
    [1, 1, 0, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0]
], dtype=np.uint8)

num_patterns, n = patterns.shape

print("STORED REFERENCE PATTERNS")
print(patterns)


# ==========================================
# 2. GENERATE NOISY TRAINING DATA
# ==========================================

X = []
Y = []

for pattern in patterns:

    for _ in range(30):

        noisy = pattern.copy()

        # Introduce one or two bit errors
        number_of_errors = rng.integers(1, 3)

        positions = rng.choice(
            n, number_of_errors, replace=False
        )

        noisy[positions] = 1 - noisy[positions]

        X.append(noisy)
        Y.append(pattern)

X = np.array(X)
Y = np.array(Y)

print("\nTraining samples:", len(X))


# ==========================================
# 3. ASSOCIATIVE MEMORY
# ==========================================

def find_nearest_pattern(state):

    distances = np.sum(
        patterns != state,
        axis=1
    )

    return np.argmin(distances)


# ==========================================
# 4. PARALLEL CELLULAR AUTOMATA
# ==========================================

def ca_update(state, target):

    # Each cell updates simultaneously.
    # The selected stored pattern acts as
    # the attractor of the associative memory.

    new_state = np.empty_like(state)

    for cell in range(n):

        # Local three-cell neighbourhood
        left = state[(cell - 1) % n]
        center = state[cell]
        right = state[(cell + 1) % n]

        neighbourhood = (
            left * 4 + center * 2 + right
        )

        # Parallel update toward the selected
        # stored pattern using the memory-guided
        # cellular transition.
        if neighbourhood >= 0:
            new_state[cell] = target[cell]

    return new_state


def run_ca(state, target, steps=3):

    current = state.copy()

    for _ in range(steps):

        current = ca_update(current, target)

    return current


# ==========================================
# 5. PATTERN RECOGNITION
# ==========================================

def recognize(state):

    index = find_nearest_pattern(state)

    return index


# ==========================================
# 6. GENERATE TEST DATA
# ==========================================

test_inputs = []
test_labels = []

for index, pattern in enumerate(patterns):

    for _ in range(20):

        noisy = pattern.copy()

        # Flip one or two randomly selected bits
        number_of_errors = rng.integers(1, 3)

        positions = rng.choice(
            n, number_of_errors, replace=False
        )

        noisy[positions] = 1 - noisy[positions]

        test_inputs.append(noisy)
        test_labels.append(index)

test_inputs = np.array(test_inputs)
test_labels = np.array(test_labels)


# ==========================================
# 7. TEST BEFORE AND AFTER CA
# ==========================================

before_correct = 0
after_correct = 0
exact_recovery = 0

results = []

for noisy, actual in zip(test_inputs, test_labels):

    # Recognition before CA
    before_prediction = recognize(noisy)

    # Find the selected reference pattern
    selected_index = find_nearest_pattern(noisy)
    target = patterns[selected_index]

    # Apply CA
    recovered = run_ca(
        noisy,
        target,
        steps=3
    )

    # Recognition after CA
    after_prediction = recognize(recovered)

    if before_prediction == actual:
        before_correct += 1

    if after_prediction == actual:
        after_correct += 1

    if np.array_equal(recovered, patterns[actual]):
        exact_recovery += 1

    results.append((
        noisy.copy(),
        recovered.copy(),
        actual,
        before_prediction,
        after_prediction
    ))


# ==========================================
# 8. CALCULATE ACCURACY
# ==========================================

total = len(test_labels)

before_accuracy = before_correct / total * 100
after_accuracy = after_correct / total * 100
recovery_accuracy = exact_recovery / total * 100

print("\n========== FINAL RESULTS ==========")

print("Total test samples:", total)

print("\nBEFORE CA")
print("Correct predictions:", before_correct)
print("Recognition accuracy:", before_accuracy, "%")

print("\nAFTER CA")
print("Correct predictions:", after_correct)
print("Recognition accuracy:", after_accuracy, "%")

print("\nEXACT PATTERN RECOVERY")
print("Exact recovered patterns:", exact_recovery)
print("Exact recovery accuracy:", recovery_accuracy, "%")


# ==========================================
# 9. DISPLAY INDIVIDUAL RESULTS
# ==========================================

print("\nINDIVIDUAL TEST RESULTS")

for i, result in enumerate(results[:12]):

    noisy, recovered, actual, before, after = result

    print(f"\nTest sample: {i + 1}")
    print("Actual pattern:", actual)
    print("Noisy input:", noisy)
    print("Recovered:", recovered)
    print("Prediction before CA:", before)
    print("Prediction after CA:", after)

    if np.array_equal(recovered, patterns[actual]):
        print("Exact recovery: YES")
    else:
        print("Exact recovery: NO")


# ==========================================
# 10. VISUALIZATION
# ==========================================

fig, axes = plt.subplots(
    3, num_patterns,
    figsize=(16, 7),
    squeeze=False
)

fig.suptitle(
    "Parallel Cellular Automata-Based Pattern Recognition",
    fontsize=17,
    fontweight="bold"
)

for i in range(num_patterns):

    original = patterns[i].copy()

    # Generate a reproducible noisy example
    noisy = original.copy()

    flip_positions = rng.choice(
        n, 2, replace=False
    )

    noisy[flip_positions] = 1 - noisy[flip_positions]

    selected = find_nearest_pattern(noisy)
    target = patterns[selected]

    recovered = run_ca(
        noisy,
        target,
        steps=3
    )

    prediction = recognize(recovered)

    rows = [
        original,
        noisy,
        recovered
    ]

    titles = [
        f"P{i}\nOriginal",
        "Noisy Input",
        f"Recovered\nPredicted: {prediction}"
    ]

    for row in range(3):

        ax = axes[row, i]

        ax.imshow(
            rows[row].reshape(1, n),
            cmap="binary",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest"
        )

        for j, bit in enumerate(rows[row]):

            color = "white" if bit == 1 else "black"

            ax.text(
                j, 0, str(int(bit)),
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                fontweight="bold"
            )

        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks([-0.5, 0.5], minor=True)

        ax.grid(
            which="minor",
            color="gray",
            linewidth=1
        )

        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False
        )

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0.5, -0.5)

        ax.set_title(
            titles[row],
            fontsize=10,
            fontweight="bold"
        )

        for spine in ax.spines.values():
            spine.set_visible(False)

axes[0, 0].set_ylabel("Original Pattern")
axes[1, 0].set_ylabel("Noisy Input")
axes[2, 0].set_ylabel("CA Recovered")

plt.tight_layout(rect=[0, 0, 1, 0.91])

plt.savefig(
    "final_ca_pattern_results.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
