"""CP-18's daily pipeline around the frozen A1 package (CP-18a: development data only).

Nothing here changes the frozen policy: every issuance is `cp17.daily.issue_day`, every check before a run is
`cp17.hostcheck`, every state is `cp17.shadow.ShadowStates`, and the store extends `cp17c.store.Store` without
weakening it. The modules:

- `plan`: the cutoff, the current target day, decision 23's catch-up and the red line (`cp25.guard`);
- `causes`: r19 §24.13's five causes of a day that is not issued;
- `store`: the live state store (the issued record, the single source), its run, not-issued, search and carry
  entries;
- `claim`: one issuance per delivery day across the runner and the local fallback (`origin/main` is the claim);
- `pipeline`: one run (the H2 gate once, the catch-up, each issuance, the record);
- `live`: the live-input payload builder (retrieval clients, the input archive, the availability judgement);
- `r7`: R7's rule wherever the pipeline verifies a bundle;
- `local`: the local fallback on the Owner's Mac (one command, a launchd agent proposed, not installed).
"""
