"""CP-17c-probe: can a GitHub-hosted runner be the issuing host? (the Owner's decision 24).

The probe measures; it changes nothing frozen. `run` executes the frozen daily chain (`cp17.daily`) under one
Node.js binary on one machine, `inputs` prepares what a runner needs from the reference runs, and `compare`
judges the runners' downloaded records against the frozen host's.
"""
