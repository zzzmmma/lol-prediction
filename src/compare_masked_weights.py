"""Compare five-seed validation results for the four masked loss weights."""
from .train_masked import (MASKED_MODELS, OUTPUT, SEEDS, prepare_extended_split,
                           summarize, validation_result)


def main():
    _, _, split = prepare_extended_split()
    groups = {}
    for variant in MASKED_MODELS:
        for coefficient in (0, 0.1, 0.3, 1):
            condition = f'lambda_{coefficient}'
            holdout = OUTPUT / condition / 'validation'
            groups[condition, variant] = [
                validation_result(holdout, variant, seed, split, coefficient)
                for seed in SEEDS
            ]
    summarize(OUTPUT / 'validation_weight_comparison.csv', groups, ('validation',))


if __name__ == '__main__':
    main()
