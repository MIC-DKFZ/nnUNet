# Contributing to nnU-Net

Thank you for your interest in contributing to nnU-Net.

nnU-Net is developed and maintained by researchers at DKFZ. There is no dedicated funding or staff for maintaining the 
repository, and development happens alongside research and teaching responsibilities. Our bandwidth for reviewing 
external contributions is therefore limited, and review times may be long.

## General principles

nnU-Net is intentionally designed to be focused, stable, and generally applicable across datasets and use cases. 
Contributions should respect this philosophy and should not introduce unnecessary complexity or specialization.

New functionality must either be generally valid across datasets and setups or convincingly benefit a large enough 
portion of the user base. We aim to avoid bloating the framework or increasing its complexity further.

## How to contribute

For larger features and refactors, please open a GitHub issue to discuss the idea before starting work. Tag
@FabianIsensee so that the discussion doesn't get missed.

To submit a contribution, fork the repository, make your changes on a branch, and open a pull request.

## Bug reports and bug fixes

Bug reports must include a minimal reproducible example. Without a repro, it is usually impossible for us to 
investigate issues.

Pull requests fixing bugs should also include a clear reproduction of the issue and an explanation of how the fix 
resolves it.

## Performance improvements

If a pull request claims performance improvements, it must include benchmarks demonstrating the effect. The benchmark 
setup must be described clearly enough for us to reproduce the results independently. We may run additional 
tests ourselves before merging.

## Contributions that are unlikely to be merged

To keep the framework maintainable and the workload manageable on our end, we deprioritize:

- dataset-specific code
- features that only apply to niche setups
- narrow custom architectures or training pipelines
- large refactorings without prior discussion
- small PRs fixing minor typos or formatting issues

## Final note

We appreciate the effort people invest in improving nnU-Net!