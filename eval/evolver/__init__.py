"""
Genetic algorithm-based prompt optimization for truth-bot.

Package layout:
    genome.py  -- Prompt genome / gene definitions and rendering
    fitness.py -- Fitness function: recall, verdict agreement, explanation quality
    ga.py      -- Genetic algorithm: selection, crossover, mutation, elitism
    runner.py  -- Execute truth-bot pipeline with custom prompts
    report.py  -- Generate human-readable fitness reports
"""
