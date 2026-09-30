# Measurement and model provenance

The release separates four evidence classes:

1. `T4_measured_input`: empirical nominal-rate NVIDIA T4 execution-time
   profiles used by the primary processor model.
2. `power_capped_T4_measured_input`: empirical NVIDIA T4 execution-time
   profiles measured after applying the declared slowdown/power-capped mode;
   these values represent the lower-rate processor service contract.
3. `T4_prototype_measurement`: prototype NVIDIA T4 inference or processing
   measurements identified by their checkpoint protocol.
4. `prototype_host_CPU_measurement`: CPU preprocessing or scheduler-runtime
   measurements, reported separately from GPU inference profiles.

The RC parameters are model parameters selected to represent the T4 operating
regime using published operating specifications. The thermal results evaluate
the conditional RC-model contract and its sensitivity; they are distinct from
synchronized physical temperature validation.
