"""CP-17-probe: the in-browser feasibility probe for A1 before the CP-17 freeze (capstone v21-r17 §19.3 item 4).

`dailyfit` runs unchanged in both runtimes, native CPython and Pyodide; it calls the landed fit code
(`cp24.member.fit_member`, `cp24.ddnn2`, `cp15.models.fit_day`, `cp16.residuals`). `payload` builds the
shipped training windows natively, `native` produces and checks the native reference, `compare`
measures deviation and selection agreement, and `measure` drives Chromium and WebKit.
"""
