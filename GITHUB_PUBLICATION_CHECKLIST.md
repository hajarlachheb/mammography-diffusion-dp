# GitHub publication checklist

The repository is published at <https://github.com/hajarlachheb/mammography-diffusion-dp>. This checklist records the
release review and provides checks for future updates.

## 1. Review the candidate locally

```bash
cd midl_public_repo_candidate_v1
git --no-pager log --stat -n 2 # review the release and subsequent updates
git status --ignored           # nothing unexpected, ignored files listed separately
git ls-files | less            # every file that would be published
```

Read, at minimum:
- `README.md`;
- `docs/*.md`;
- `REPOSITORY_AUDIT.md` (open actions A1–A6);
- `SECURITY_AUDIT.md`;
- `DATA_REDISTRIBUTION_AUDIT.md`;
- `THIRD_PARTY_LICENSES.md`.

## 2. Complete the author actions (REPOSITORY_AUDIT.md A1–A6)

- [x] **A1, author metadata.** `CITATION.cff` contains the confirmed name, affiliation and `repository-code`;
  `NOTICE` contains the copyright holder; `date-released` is set to the public release date.
- [x] **A1, commit author.** The placeholder identity was replaced with the confirmed name and GitHub noreply address:
  ```bash
  git config user.name "Hajar Lachheb"
  git config user.email "<id>+<username>@users.noreply.github.com"
  git commit --amend --reset-author --no-edit
  ```
- [x] **A2, license.** The author confirmed Apache-2.0 and that no CC-NET source was copied verbatim.
- [x] **A3, data terms.** The author confirmed the CBIS-DDSM terms (TCIA) and the appropriateness of pointing to
  `mammo_dp/dataset16062024`.
- [x] **A4, base model.** The author chose to keep `Manojb/stable-diffusion-2-1-base`; its mirror provenance is
  documented in `THIRD_PARTY_LICENSES.md`.
- [ ] **A5, citation.** Add the manuscript citation only when it exists.
- [x] **A6, institutional approval.** The author reports that required institutional approval has been obtained.

## 3. Re-verify after any edit

```bash
python -m venv /tmp/mdpd && source /tmp/mdpd/bin/activate
pip install -r requirements.txt && pip install -e ".[dev]"
pytest                                   # expect: all passed
python scripts/reproduce_tables.py && python scripts/reproduce_figures.py && git status   # expect: no changes
find . -name __pycache__ -prune -exec rm -rf {} +; rm -rf src/*.egg-info .pytest_cache
grep -rnE "/(media|home|mnt)/|@[a-z0-9-]+\.(com|org|net)" -- $(git ls-files) | grep -v -e "tests/test_paths_and_cli.py" -e GITHUB_PUBLICATION_CHECKLIST.md   # expect: nothing
```

## 4. Published repository and future updates

The repository is public; do not create a second repository or repeat the initial-visibility steps. For a future update:

1. **Push the reviewed change** to the existing `main` branch:
   ```bash
   git push origin main
   ```
2. **Confirm the CI workflow passes** in the Actions tab ("tests").
3. **Check the rendered pages** and confirm README, figures, docs and links are correct.
4. **Optional, later:**
   - tag a release once the manuscript version is final;
   - archive it with Zenodo for a DOI;
   - add that DOI to `CITATION.cff` and the README.

## 5. Never add

- Datasets or image patches (CBIS-DDSM, BCDR, EUCanImage), synthetic images, or example-image figures.
- Model weights or checkpoints, unless a separate review approves them (REPOSITORY_AUDIT.md R3).
- Run directories, training logs, per-image prediction files and manifests with file names or patient IDs.
- `.env`, credentials, SSH or VPN files, and server paths.
