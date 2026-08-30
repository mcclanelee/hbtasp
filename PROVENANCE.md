# Measurement and model provenance

The release separates four evidence classes:

1. `T4_measured_input`: empirical NVIDIA T4 execution-time profiles used by the
   primary processor model.
2. `degraded_T4_simulation_input`: the declared slower second-processor service
   profile used to represent heterogeneous capacity.
3. `T4_prototype_measurement`: prototype NVIDIA T4 inference or processing
   measurements identified by their checkpoint protocol.
4. `prototype_host_CPU_measurement`: CPU preprocessing or scheduler-runtime
   measurements, reported separately from GPU inference profiles.

The RC parameters are model parameters selected to represent the T4 operating
regime using published operating specifications. The thermal results evaluate
the conditional RC-model contract and its sensitivity; they are distinct from
synchronized physical temperature validation.
