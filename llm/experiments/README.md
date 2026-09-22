# Experiment records

Each experiment gets one Markdown report named `<ID>_<short-description>.md`.
Keep the result table near the top, followed by:

1. identity: date, commit, environment, GPU, model/data revisions;
2. variable: exactly what changed and the baseline;
3. command and checkpoint/scheduler state;
4. evaluation set IDs, counts, exclusions, and leakage notes;
5. metrics with definitions and uncertainty;
6. conclusion: adopt, reject, or evidence insufficient;
7. next experiment.

Store machine-readable per-sample results beside the report as JSON. Never
replace an earlier report; append a new report for each run so later work can
reuse both the conclusions and the reasons for rejecting alternatives.
