# Fast battle evaluator support

This directory contains the worker, observation encoder, actor inference, and process-management code used by the [main battle pipeline](../team_completion_pipeline/FAST_EVALUATOR_INTEGRATION.md). The integrated configuration uses six persistent workers, 24 concurrent battles per worker, shared batching, and one numerical CPU thread.

The existing local pilot environment is required: its `corpus`, `matchup_db`, and `pool` modules, the pinned VGC-Bench source, the pinned Showdown runtime, and the verified `bc_100.zip` checkpoint. Their locations are specified in the source and match the existing Mask Lab installation. This PR does not bundle third-party runtimes, trained weights, or private databases.

`run.py` also retains the original isolated experiment harness. Optional experimental inference modes are not enabled by the integrated pipeline. See the pipeline report for the tested configuration, commands, limitations, and measured throughput.
