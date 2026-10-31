# The Owner's decisions after CP-17 — 2026-10-28

**Authority.** The Owner, 2026-10-28, on the Orchestrator's summary of CP-17's return
([landing record](cp-17-landing-2026-10-28.md)): "אני מאשר את כל ההמלצות שלך". With it: "יש לך
אישור מלא עם כל ההשעיות וכל מה שאתה צריך".

**Why a new record.** CP-17's freeze pins `pre-cp-17-rulings-2026-10-25.md` by hash
(`src/cp17/tools/freeze.py`), so decisions 1–12 stay in that file unchanged. Decisions from 13 on are
recorded here.

These decisions answer the items CP-17's freeze record §11 left open. They were taken without any
4.7T outcome, since none exists. The anchor amendment r18 ratifies them before 4.7T opens. Until then
they bind the programme as decisions.

13. **C2, partial days.** A fit that emits every hour the frozen design can forecast is a successful
    fit on those hours. It is displayed as partial, with the missing hours named. A day with no
    forecastable hour is a failed issuance.
    - **The reason.** 43 of CP-26's 2,078 scored development days have incomplete issuance keys,
      mostly the days after a DST change. CP-26 and CP-28 scored them on CP-25's keys, and so
      evaluated A1. CP-17's brief had required every delivery hour, which would make every such day a
      failed issuance.
14. **G1, shadow issuances.** The fallback policies issue every day under the issuing host as shadow
    issuances, which are never published, so that their own interval buffers stay current.
    - **The policies:** v4 and v2. B0 trains nothing, and its raw-error buffer comes from published
      prices.
    - **The cost:** about 100 s a day, within the 15-minute ceiling.
15. **The fallback chains stay as decisions 11 and 12 set them.** The Owner asked whether a fallback's
    own fallback was ever needed. If it was not, he would remove them.
    - **What the record shows.** In CP-26's 2,078 scored development days:
      - D2 failed on 18, and v4 was native on all 18;
      - v4, v2 and B0 never failed on their own;
      - on 10 days every policy, B0 included, had no forecastable hour, so no fallback could help.
    - **Why they stay.** Each second fallback guards a live cause the archive cannot show:
      - **B0 last:** a late or missing load forecast. A1, v4 and v2 all need it, and B0 needs only
        prices. CP-25's margin table shows that a one-hour delay reaches the issue time on every day.
        That is the risk accepted for measurement in CP-18.
      - **v2 after v4:** GFS missing inside the training window (G2). A1 and v4 need the weather
        history, and v2 does not.
    - **Removing them would not shorten G1.** v4 and v2 issue in the shadow anyway, as the first
      fallbacks of the two causes.
16. **C1, the label wording.** Released errors come from complete delivery days ≤ D−2 (§2). Training
    targets are the delivery days before D whose prices were published strictly before the issue
    time, so D−1 is a training target. The landed code and CP-26's parity already follow this. Only
    the wording of CP-17's brief was wrong.
17. **G2 is recorded as the frozen behaviour.** GFS missing in A1's training window, but not at the
    issue time, fails A1's fit. v4 fails for the same reason, and v2 issues.
18. **F1 is disclosed, and τ_eq is unchanged.**
    - At the DST origin 2025-10-26, A1 under the issuing host differs from native by up to
      0.028 EUR/MWh, with every selection identical.
    - τ_eq = 0.01 stays the visitor's tolerance against the issued artifact, which the browsers
      reproduce bit for bit.
    - The native parity with CP-26 stays the check of code identity.
19. **A short re-freeze round, CP-17b, before r18.** It:
    - implements decisions 13 and 14;
    - fixes the review's notes N1–N7;
    - moves the freeze's pins of governance documents to frozen copies, so that r18 and later
      records do not break `freeze.verify`.

    r18 then ratifies the terms with CP-17b's freeze identity. It is done under the Owner's
    suspension, granted with these decisions, and with an independent plan review, before
    2026-12-04.
