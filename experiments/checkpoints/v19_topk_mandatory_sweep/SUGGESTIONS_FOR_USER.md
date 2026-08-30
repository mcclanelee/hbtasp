# Suggestions for User Review

## Suggestion SUG-001

Category: experiment

Observation: Endpoint-only Top-1 versus Top-4 evidence could not exclude a dominating Top-2 or Top-3 policy. The cardinality sweep showed that this check materially strengthens the mandatory/optional design claim.

Trigger: A paper chooses a fixed number of protected regions or tasks while claiming a coverage--schedulability trade-off.

Suggested rule or change: Sweep every feasible protection cardinality under an otherwise frozen protocol before calling the selected cardinality a favorable trade-off.

Recommended action: accept

Scope: real-time regional-inference papers

Automation level: ask-user

Risk: low

Reason: It prevents an unsupported claim that an endpoint configuration is near a Pareto knee.

User decision: PENDING

