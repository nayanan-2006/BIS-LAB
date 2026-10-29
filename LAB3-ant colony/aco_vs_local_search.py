import numpy as np
import matplotlib.pyplot as plt

SEED = 42
N_CITIES = 15
N_ANTS = 25
ITERATIONS = 100
CHANGE_EVERY = 25
ALPHA = 1.0
BETA = 5.0
EVAPORATION = 0.3

rng = np.random.default_rng(SEED)
cities = rng.uniform(0, 100, (N_CITIES, 2))
base_dist = np.linalg.norm(cities[:, None, :] - cities[None, :, :], axis=2)
np.fill_diagonal(base_dist, 0)


def tour_cost(route, distances):
    return sum(distances[route[i], route[(i + 1) % len(route)]] for i in range(len(route)))


def relocate_local_search(route, distances):
    """Simplified remove-and-reinsert search (NOT the paper's full US operator)."""
    best = list(route)
    best_cost = tour_cost(best, distances)
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            reduced = best[:i] + best[i + 1:]
            for j in range(len(best)):
                candidate = reduced[:j] + [best[i]] + reduced[j:]
                candidate_cost = tour_cost(candidate, distances)
                if candidate_cost < best_cost - 1e-9:
                    best, best_cost = candidate, candidate_cost
                    improved = True
                    break
            if improved:
                break
    return best


def make_dynamic_matrix(base, random_generator):
    """Traffic changes on a fraction of directed arcs; distances can become asymmetric."""
    changed = base.copy()
    mask = random_generator.random(base.shape) < 0.25
    np.fill_diagonal(mask, False)
    factors = random_generator.uniform(1.0, 3.0, base.shape)
    changed[mask] *= factors[mask]
    return changed


def solve_aco(distances, seed, use_local_search=False):
    random_generator = np.random.default_rng(seed)
    n = len(distances)
    pheromone = np.ones((n, n), dtype=float)
    np.fill_diagonal(pheromone, 0)
    best_route, best_cost = None, float('inf')
    history = []
    for iteration in range(ITERATIONS):
        iteration_best_route, iteration_best_cost = None, float('inf')
        for _ in range(N_ANTS):
            start = int(random_generator.integers(n))
            route = [start]
            unvisited = set(range(n)) - {start}
            while unvisited:
                current = route[-1]
                options = sorted(unvisited)
                weights = np.array([
                    pheromone[current, j] ** ALPHA *
                    (1.0 / max(distances[current, j], 1e-12)) ** BETA
                    for j in options
                ])
                weights /= weights.sum()
                nxt = int(random_generator.choice(options, p=weights))
                route.append(nxt)
                unvisited.remove(nxt)
            cost = tour_cost(route, distances)
            if cost < iteration_best_cost:
                iteration_best_route, iteration_best_cost = route, cost
        if iteration_best_cost < best_cost - 1e-9:
            candidate = iteration_best_route
            if use_local_search:
                candidate = relocate_local_search(candidate, distances)
            candidate_cost = tour_cost(candidate, distances)
            if candidate_cost < best_cost:
                best_route, best_cost = candidate.copy(), candidate_cost
        # MMAS-inspired pheromone update with lower and upper bounds.
        pheromone *= (1 - EVAPORATION)
        upper = 1.0 / (EVAPORATION * best_cost)
        lower = upper / (2 * n)
        for a, b in zip(best_route, best_route[1:] + best_route[:1]):
            pheromone[a, b] += 1.0 / best_cost
        pheromone = np.clip(pheromone, lower, upper)
        np.fill_diagonal(pheromone, 0)
        history.append(best_cost)
    return best_route, best_cost, history


def plot_route(ax, route, cost, title):
    closed = route + [route[0]]
    xy = cities[closed]
    ax.plot(xy[:, 0], xy[:, 1], '-o', linewidth=1.5)
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), xytext=(4, 4), textcoords='offset points')
    ax.set_title(f'{title}\nCost = {cost:.2f}')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(alpha=0.25)


def main():
    # Static TSP: both methods solve exactly the same city/distance instance.
    distances = base_dist.copy()
    plain_route, plain_cost, plain_history = solve_aco(distances, SEED, False)
    hybrid_route, hybrid_cost, hybrid_history = solve_aco(distances, SEED, True)

    print('STATIC TSP: ACO vs ACO + Local Search')
    print('Note: relocation search approximates, but does not reproduce, the paper\'s US operator.')
    print('Standard ACO route:', plain_route + [plain_route[0]])
    print(f'Standard ACO distance: {plain_cost:.2f}')
    print('ACO + Local Search route:', hybrid_route + [hybrid_route[0]])
    print(f'ACO + Local Search distance: {hybrid_cost:.2f}')
    print(f'Distance improvement: {plain_cost - hybrid_cost:.2f}')
    print(f'Percentage improvement: {100 * (plain_cost - hybrid_cost) / plain_cost:.2f}%')

    # Figure 1: routes for the SAME cities and distance matrix.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_route(axes[0], plain_route, plain_cost, 'Standard ACO')
    plot_route(axes[1], hybrid_route, hybrid_cost, 'ACO + Local Search')
    fig.suptitle('Route Comparison: ACO vs ACO + Local Search')
    fig.tight_layout()
    plt.show()

    # Figure 2: best-so-far distance over iterations.
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, ITERATIONS + 1), plain_history, label='Standard ACO', linewidth=2)
    plt.plot(range(1, ITERATIONS + 1), hybrid_history, label='ACO + Local Search', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best tour distance')
    plt.title('Convergence: ACO vs ACO + Local Search')
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("name : Nayana N \n USN 1WN24CS171")


if __name__ == '__main__':
    main()
