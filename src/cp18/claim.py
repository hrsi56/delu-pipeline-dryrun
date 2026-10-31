"""One issuance per delivery day across the runner and the local fallback: the claim.

**The rule.** The day's issued record on the product repository's `main` is the claim.
- **Fetch before issuing.** Before each day, the pipeline fetches `main` and checks out exactly that commit (detached,
  on a clean tree), then reloads and verifies the store. A day already decided there is never issued again.
- **Push after issuing.** The day's records are committed on top of that commit, with the secret guard, the leak scan
  and the publication guard running as git hooks (`core.hooksPath` must resolve to the repository's `.githooks`, or
  nothing is committed), and pushed as a fast-forward of `main`. Nothing is ever force-pushed.
- **The loser of a push race discards its own issuance.** If `main` moved while the day was being issued, the push is
  refused as a non-fast-forward. The pipeline then fetches, checks out the new `main` (its own commit, which no ref
  names, is dropped), and stops the run: the other host's record is the day's issuance, and nothing is replaced,
  rescored or relabelled. A push refused for any other reason (a guard, the network) is not a lost race: the run
  fails closed and publishes nothing.

The git commands run in the pipeline's own checkout: a fresh one on the runner, a dedicated clone on the Owner's Mac.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

IDENTITY = {'GIT_AUTHOR_NAME': 'delu-daily', 'GIT_AUTHOR_EMAIL': 'delu-daily@users.noreply.github.com',
            'GIT_COMMITTER_NAME': 'delu-daily', 'GIT_COMMITTER_EMAIL': 'delu-daily@users.noreply.github.com'}


class ClaimError(RuntimeError):
    """The checkout cannot be synchronized or published safely; the run publishes nothing."""


class Claim:
    def __init__(self, repo, remote: str = 'origin', branch: str = 'main', env: dict | None = None):
        self.repo, self.remote, self.branch = Path(repo), remote, branch
        self.env = {**os.environ, **IDENTITY, 'GIT_TERMINAL_PROMPT': '0', 'GIT_PAGER': 'cat', 'GIT_EDITOR': 'true',
                    **(env or {})}

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        done = subprocess.run(['git', '-C', str(self.repo), *args], capture_output=True, text=True, env=self.env)
        if check and done.returncode != 0:
            raise ClaimError(f'git {args[0]} failed (exit {done.returncode}): {done.stderr.strip()[-600:]}')
        return done

    def remote_ref(self) -> str:
        return f'refs/remotes/{self.remote}/{self.branch}'

    def head(self) -> str:
        return self.git('rev-parse', 'HEAD').stdout.strip()

    def hooks_active(self) -> bool:
        """`core.hooksPath` resolves to this checkout's `.githooks` (relative or absolute)."""
        value = self.git('config', '--get', 'core.hooksPath', check=False).stdout.strip()
        if not value:
            return False
        path = Path(value) if Path(value).is_absolute() else self.repo / value
        return path.resolve() == (self.repo / '.githooks').resolve()

    def fetch(self) -> str:
        self.git('fetch', '--quiet', '--no-tags', self.remote,
                 f'+refs/heads/{self.branch}:{self.remote_ref()}')
        return self.git('rev-parse', self.remote_ref()).stdout.strip()

    def sync(self) -> str:
        """Fetch `main` and check out exactly that commit, detached, on a clean tree."""
        if self.git('status', '--porcelain', '--untracked-files=all').stdout.strip():
            raise ClaimError('the pipeline checkout has uncommitted changes; it never issues over them')
        tip = self.fetch()
        self.git('checkout', '--quiet', '--detach', tip)
        return tip

    def moved(self) -> bool:
        """Whether `main` on the remote has moved past this checkout's base since the last sync."""
        before = self.git('rev-parse', self.remote_ref()).stdout.strip()
        return self.fetch() != before

    def publish(self, paths: list[str], message: str) -> dict:
        """Commit `paths` on the synchronized commit and fast-forward `main`. Returns {'pushed': sha} or, when another
        host moved `main` first, {'lost': True, 'theirs': sha} after discarding this commit."""
        if not self.hooks_active():
            raise ClaimError('core.hooksPath does not point at the checkout\'s .githooks: nothing is committed '
                             'without the secret guard and the leak scan')
        base = self.git('rev-parse', self.remote_ref()).stdout.strip()
        if self.head() != base:
            raise ClaimError('the checkout is not on the synchronized main; nothing is committed')
        self.git('add', '--', *paths)
        if not self.git('diff', '--cached', '--name-only').stdout.strip():
            return {'pushed': None, 'nothing': True}
        done = self.git('commit', '--quiet', '-m', message, check=False)
        if done.returncode != 0:
            self.git('reset', '--quiet')                  # unstage only; the files stay for the run's own record
            raise ClaimError(f'the commit was refused (a guard?): {done.stderr.strip()[-600:]}')
        mine = self.head()
        pushed = self.git('push', '--quiet', '--porcelain', self.remote, f'HEAD:refs/heads/{self.branch}', check=False)
        if pushed.returncode == 0:
            self.fetch()
            return {'pushed': mine}
        theirs = self.fetch()
        ancestor = self.git('merge-base', '--is-ancestor', base, theirs, check=False).returncode == 0
        if theirs != base and ancestor:
            self.git('checkout', '--quiet', '--detach', theirs)     # this commit, named by no ref, is dropped
            return {'lost': True, 'theirs': theirs, 'discarded': mine}
        self.git('checkout', '--quiet', '--detach', base)
        raise ClaimError(f'the push was refused and main did not move (a guard or the network): '
                         f'{pushed.stderr.strip()[-600:]}')
