"""CP-17: the frozen product policy on one code path (capstone v21-r17 §19.2, §24.9).

The product modules run unchanged under native CPython and under the pinned Pyodide runtime, in Node.js
(the issuing host, the Owner's decision 10) and in the browser (§19.3). Their only input is a shipped
payload (`cp17.schema`):

- `dailyfit` fits one delivery day of A1 (DDNN-2 alone), v2, v5 or B0 from its payload, through the
  landed model code;
- `issue` applies the frozen failure policy and writes the issued artifact;
- `scoring` implements §19.5's definitions and the recomputation check;
- `attribution` implements §19.7's methods by component class;
- `retrain` is §19.3's comparison and its status rule (τ_eq, the Owner's decision 9);
- `schema` holds the payload schemas and their validator.

`cp17.tools` holds the native-only drivers (payload construction, CP-26's stored state, the hosts and the
records); it never ships.
"""
