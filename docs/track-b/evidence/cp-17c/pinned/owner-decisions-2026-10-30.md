# The Owner's decisions on daily operation and the road to the live product — 2026-10-30

**Authority.** The Owner, in the session of 2026-10-29 to 2026-10-30, after the Orchestrator explained
CP-18 and where the daily fit runs. His words, quoted in order:
- "אנחנו יכולים לעשות שהתזמון יהיה לא קשיח. לאף אחד לא באמת אכפת אם פרסמנו את הנתונים בשעה
  מסויימת או אחרי שנתוני האמת כבר יצאו אפילו." ("We can make the timing soft. Nobody really cares
  whether we published at a given hour, or even after the real data came out.")
- "אפילו מצידי לא לרשות שהיא בוצעה באיחור. כל עוד היא קרתה, היא קרתה, לא קרתה לא קרתה." ("As far
  as I'm concerned, don't even record that it ran late. If it happened, it happened; if it didn't,
  it didn't.")
- "אימון בדיקת ימים חסרים כל פעם זה בסדר ומקובל אבל צריך להגביל את זה שלא ייקרה מצב שפתאום הוא
  מנסה להריץ 20 יום ונתקע ומצטברים לו ימים ולמחרת 21 22 23 וככה לנצח לא יאומן לעולם." ("Catching up
  missed days on each run is fine, but it must be bounded, so that it never tries to run 20 days,
  gets stuck, and the backlog grows to 21, 22, 23 until it never trains again.")
- "אם הוא עובר חלק, תחליט לבד בין לינוקס למק" ("If it passes cleanly, choose between Linux and Mac
  yourself"), then: r19, a short re-freeze, and "פרסום קומפלט, עם מוצר חי ועם כל מה שעשינו עד כה ולא
  פורסם" ("a complete publication, with a live product and everything done so far and not yet
  published").
- "רוץ באופן עצמאי כל עוד אין בעיות חדשות שצצות החלטות משמעותיות אני מאשר לך לפעול עצמאית לחלוטין"
  ("Run independently. Unless new problems or significant decisions come up, I authorize you to act
  fully on your own.")

**Why a new record.** Decisions 1–19 are bound by CP-17b's frozen copies, and 20–21 are in
`owner-decisions-2026-10-29.md`. These continue the numbering. None was taken with any 4.7T outcome,
since none exists, and none changes 4.7T's registered terms.

22. **The publication time is not part of the product's claim.**
    - **The information cutoff stays as frozen:** D−1 11:00 Europe/Berlin, strictly before (§24.4;
      CP-17b's freeze record §§4 and 6). A run may happen at any time after it. Its payload holds only
      inputs published before the cutoff, so a run made hours later issues the same forecast.
    - **A delivery day is issued or not issued.** There is no "late" status and no "late" label.
    - **What stays recorded:** each issuance's actual run time, which §16.2 requires in the artifact's
      lineage. The product panel states when the last training ran. No public wording claims that a
      forecast was published before the auction, or before its outcome.
    - **What r19 changes:** §24.4's sentence on the live product's 12:00 deadline and late forecasts,
      and §16.4's CP-19 count of "timely" forecasts, which becomes a count of issued and not-issued
      days. `AGENTS.md`'s daily-update exception needs no change: a day that is not issued is shown as
      not issued, the panel shows the date of its last training, and nothing is labelled timely.
23. **Missed days are caught up, with a hard bound.**
    - **Each run** issues, in chronological order, the missed delivery days within a lookback of three
      days before the current target day, then the current target day: **at most four issuances per
      run.**
    - **A day older than the lookback is never issued,** and stays "not issued". The backlog therefore
      never exceeds three days, and every run reaches the current day.
    - **Each catch-up** uses only inputs published before that day's own cutoff, as the frozen rules
      require, and never replaces an issued forecast.
    - **The bound in time:** each fit keeps its frozen 15-minute ceiling, so a run takes at most about an
      hour.
    - CP-18 may shorten the lookback after measuring the runtime. It may not remove the bound.
24. **The issuing host may move to a GitHub-hosted runner.**
    - **A probe first, CP-17c-probe,** on development data only: on a Linux x64 runner and on a macOS
      Apple-silicon runner, it checks whether the daily fit reproduces the frozen issuing host bit for
      bit, for A1 and the three shadows, within the frozen time and memory ceilings.
    - **If a runner passes cleanly,** the Orchestrator chooses between Linux and macOS. r19 and a short
      re-freeze, CP-17c, then pin that host before 4.7T opens.
    - **If neither passes cleanly,** the Owner's Mac stays the issuing host. Any other outcome is a new
      decision, brought to the Owner.
    - **Until then** the frozen host is unchanged. Homebrew's `node` is pinned on the Owner's machine
      (2026-10-30), so an upgrade cannot replace the pinned binary.
25. **The road to the live product, run by the Orchestrator.**
    - **The sequence:** CP-17c-probe → the host choice → r19 → CP-17c → CP-18's preparation → 4.7T →
      CP-18's activation → one complete publication.
    - **The complete publication** carries the live product and everything done and not yet published:
      CP-26, CP-28, CP-17 and CP-17b, r18, 4.7T's result and the live product. It replaces PRES-5 as a
      separate step.
    - **The Owner's authority** covers the Lockdown suspensions each named step needs:
      - `capstone_v21.md` and its amendment record for r19;
      - the re-freeze;
      - the publication under PUBLISH_RULES.

      The Orchestrator runs these steps without asking again.
    - **What still comes back to the Owner:**
      - a new problem, or a significant decision these records do not settle;
      - his declaration before 4.7T opens (§24.2; r18 §24.12), which only he can make. The retrieval
        approval is shown to him with it;
      - entering the pipeline's credentials as GitHub secrets, if the host moves. Agents never handle
        credential values;
      - a failed or inconclusive 4.7T. Decision 21 keeps v1 as the product, and what the publication
        then shows is a new decision.
