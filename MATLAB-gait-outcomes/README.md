# MATLAB gait outcomes

```matlab
addpath('/path/to/MATLAB-gait-outcomes');
ResultStruct = GaitOutcomesTrunkAccFuncIH( ...
    inputData, FS, LegLength, WindowLen, false, false);
```

`inputData`: N × 3 acceleration matrix in V, ML, AP order (m/s²).
`FS`: sampling frequency (Hz). `LegLength`: leg length (m).
`WindowLen`: analysis window length in samples.

With `ApplyRealignment = true`, use N × 4 input in time, ML, V, AP order.
`ApplyRemoveSteps = true` removes the first and last 30 detected steps.

Stride-time search range: **0.4–4.0 s**, updated in all three functions that define it.
Main scripts, duplicate copies, batch/export scripts, and data files are excluded.
Original function headers and calculation methods are retained.

Validated with MATLAB R2024a on a 20 s acceleration signal (51 finite outputs).

Requirements: MATLAB, Signal Processing Toolbox, and Statistics and Machine Learning Toolbox.
Sensor realignment also uses Optimization Toolbox.

Optional dependencies absent from the supplied source: `FalseNearestNeighborsSR`
(automatic embedding dimension) and `kdtree_build`, `kdtree_k_nearest_neighbors`,
`kdtree_delete` (Rosenstein state spaces with more than 10,000 rows).

`GaitOutcomesTrunkAccFuncIH_fast` is the supplied convergence-analysis variant;
it omits Rosenstein estimation. `GaitVariabilityAnalysisIH_WithoutTurns` retains
the original unresolved `data` input in its realignment branch.
