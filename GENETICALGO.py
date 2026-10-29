import random
import math


# -----------------------------------------
# 1. Calculate distance between two cities
# -----------------------------------------

def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2

    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# -----------------------------------------
# 2. Calculate total distance of a route
# -----------------------------------------

def total_distance(route, cities, salesmen):

    n = len(cities)

    # Split the chromosome into routes
    route_size = n // salesmen

    total = 0

    start = 0

    for s in range(salesmen):

        end = start + route_size

        # Last salesman gets remaining cities
        if s == salesmen - 1:
            end = n

        subroute = route[start:end]

        if len(subroute) > 0:

            # Starting city = city 0
            previous = 0

            for city in subroute:
                total += distance(cities[previous], cities[city])
                previous = city

            # Return to starting city
            total += distance(cities[previous], cities[0])

        start = end

    return total


# -----------------------------------------
# 3. Greedy population initialization
# -----------------------------------------

def greedy_route(cities):

    n = len(cities)

    unvisited = list(range(1, n))

    route = []

    current = 0

    while unvisited:

        nearest = min(
            unvisited,
            key=lambda city: distance(cities[current], cities[city])
        )

        route.append(nearest)

        current = nearest

        unvisited.remove(nearest)

    return route


def create_initial_population(cities, population_size):

    population = []

    # First route using greedy strategy
    greedy = greedy_route(cities)
    population.append(greedy)

    # Remaining routes are random
    for _ in range(population_size - 1):

        route = list(range(1, len(cities)))

        random.shuffle(route)

        population.append(route)

    return population


# -----------------------------------------
# 4. Fitness function
# -----------------------------------------

def fitness(route, cities, salesmen):

    d = total_distance(route, cities, salesmen)

    return 1 / d


# -----------------------------------------
# 5. Elitist selection
# -----------------------------------------

def selection(population, cities, salesmen):

    population.sort(
        key=lambda r: total_distance(r, cities, salesmen)
    )

    # Keep the best individual
    elite = population[0]

    return elite


# -----------------------------------------
# 6. PMX crossover
# -----------------------------------------

def pmx(parent1, parent2):

    size = len(parent1)

    child = [None] * size

    # Choose crossover points
    left, right = sorted(
        random.sample(range(size), 2)
    )

    # Copy section from parent1
    child[left:right] = parent1[left:right]

    # Mapping process
    for i in range(left, right):

        if parent2[i] not in child:

            position = i

            while left <= position < right:

                value = parent1[position]

                position = parent2.index(value)

            child[position] = parent2[i]

    # Fill remaining positions
    for i in range(size):

        if child[i] is None:

            child[i] = parent2[i]

    return child


# -----------------------------------------
# 7. Inverse mutation
# -----------------------------------------

def inverse_mutation(route):

    route = route[:]

    i, j = sorted(
        random.sample(range(len(route)), 2)
    )

    route[i:j] = reversed(route[i:j])

    return route


# -----------------------------------------
# 8. 2-opt local search
# -----------------------------------------

def two_opt(route, cities, salesmen):

    best = route[:]

    best_distance = total_distance(
        best,
        cities,
        salesmen
    )

    improved = True

    while improved:

        improved = False

        for i in range(len(best) - 1):

            for j in range(i + 1, len(best)):

                new_route = best[:]

                # Reverse part of route
                new_route[i:j] = reversed(
                    new_route[i:j]
                )

                new_distance = total_distance(
                    new_route,
                    cities,
                    salesmen
                )

                if new_distance < best_distance:

                    best = new_route

                    best_distance = new_distance

                    improved = True

        # Continue until no improvement

    return best


# -----------------------------------------
# 9. Improved Genetic Algorithm
# -----------------------------------------

def genetic_algorithm(
        cities,
        salesmen,
        population_size=80,
        generations=500
):

    population = create_initial_population(
        cities,
        population_size
    )

    best_route = None
    best_distance = float("inf")

    history = []

    for generation in range(generations):

        # Sort population
        population.sort(
            key=lambda r:
            total_distance(
                r,
                cities,
                salesmen
            )
        )

        # Current best
        current_best = population[0]

        current_distance = total_distance(
            current_best,
            cities,
            salesmen
        )

        # Update global best
        if current_distance < best_distance:

            best_distance = current_distance

            best_route = current_best[:]

        history.append(best_distance)

        # New population
        new_population = []

        # Elitism
        new_population.append(
            current_best[:]
        )

        while len(new_population) < population_size:

            # Select parents
            parent1 = random.choice(
                population[:population_size // 2]
            )

            parent2 = random.choice(
                population[:population_size // 2]
            )

            # Crossover
            if random.random() < 0.9:

                child = pmx(
                    parent1,
                    parent2
                )

            else:

                child = parent1[:]

            # ---------------------------------
            # Paper: 20% chromosomes → 2-opt
            # Remaining → inverse mutation
            # ---------------------------------

            if random.random() < 0.20:

                child = two_opt(
                    child,
                    cities,
                    salesmen
                )

            else:

                child = inverse_mutation(
                    child
                )

            new_population.append(child)

        population = new_population

    return best_route, best_distance, history


# =========================================
# MAIN PROGRAM
# =========================================

cities = [
    (10, 20),
    (30, 40),
    (50, 20),
    (60, 60),
    (20, 70),
    (80, 30),
    (90, 70),
    (40, 80),
    (15, 50),
    (70, 10)
]

salesmen = 2

best_route, best_distance, history = genetic_algorithm(
    cities,
    salesmen,
    population_size=80,
    generations=500
)

print("Best Route:")
print(best_route)

print("\nMinimum Total Distance:")
print(best_distance)