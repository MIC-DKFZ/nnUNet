verify all ranks have the same initial parameters

dice loss needs all_gather

checkpoint save fix key dicts (network.)


verify that all_reduce does not increase gradient magnitude too much. Maybe divide result by world size
(average instead of sum)?


save debug information? 