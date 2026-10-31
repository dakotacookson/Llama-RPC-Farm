# installer/ — three-layer testable installer architecture (req #3, task t_bd9422d1).
#
# Split (§10 of the design doc):
#   env.py     — SystemInfo value + injection interfaces (RunCmd / FS) + real impls
#   probe.py   — PURE decision logic: probe(state) -> Plan (no I/O beyond interfaces)
#   execute.py — dumb runner consuming Plan steps (real fs / real run in prod)
#
# The rule that makes this testable: probe.py performs ZERO subprocess/os.system
# calls and zero direct disk access — it only interprets data that was captured
# through the injected interfaces in env.py. Hermetic tests feed pyfakefs + a
# stub run() and assert on the emitted plan.