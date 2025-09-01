import pygad
from scripts.constants import NUM_FEATURES, NUM_ACTIONS, NUM_GENERATIONS, NUM_PARENTS_MATING, SOL_PER_POP, \
    MUTATION_PERCENT_GENES, GENE_SPACE_LOW, GENE_SPACE_HIGH, NUM_TESTING_STEPS
from scripts.snake import SnakeGame, fitness_func
from scripts.utils import select_action

num_genes = NUM_FEATURES * NUM_ACTIONS

ga_instance = pygad.GA(
    num_generations=NUM_GENERATIONS,
    num_parents_mating=NUM_PARENTS_MATING,
    fitness_func=fitness_func,
    sol_per_pop=SOL_PER_POP,
    num_genes=num_genes,
    mutation_percent_genes=MUTATION_PERCENT_GENES,
    mutation_type="random",
    gene_space={'low': GENE_SPACE_LOW, 'high': GENE_SPACE_HIGH}
)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"\nBest solution fitness: {solution_fitness}")


# Testing best agent
test_game = SnakeGame()
s = test_game.reset()
done = False
steps = 0
print(f"Starting test: snake={test_game.snake}, apples={test_game.food}")
while not done and steps < NUM_TESTING_STEPS:
    act = select_action(s, solution)
    s, r, done = test_game.step(act)
    steps += 1
    print(
        f"Step {steps} | Action: {act} | Reward: {r} | Score: {test_game.score} | Snake: {test_game.snake} | Remaining apples: {test_game.food}")

if not test_game.food:
    print("\nAll apples eaten!")
else:
    print("\nGame ended. Apples remaining:", test_game.food)

