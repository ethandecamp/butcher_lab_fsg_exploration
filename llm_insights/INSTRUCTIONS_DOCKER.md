# Running this in a container

This guide assumes you have **never used Docker** and are not a programmer. Every step
gives you one command to copy, and tells you what a correct result looks like. If
something goes wrong, jump to [§8, the failure table](#8-when-something-goes-wrong).

**What you are about to run.** A program that proposes claims about Dan Pearce's
fluid-solid-growth simulation of a developing heart valve, and then tries to *refute*
each one by running executable tests against the simulation output. No language model
decides whether a claim is true; that is arithmetic, done in code.

**What step 6 does not need.** No internet, no API key, no account, no language model.
It replays a recorded set of claims and re-runs every verification live. That is
deliberate: it proves the container works before you configure anything.

---

## 1. Install Docker Desktop

Go to <https://www.docker.com/products/docker-desktop/>, download the version for your
computer, and install it the way you would install any other application.

Then **open Docker Desktop and leave it running.** This matters more than it sounds:
the `docker` commands below talk to a background service that only exists while Docker
Desktop is open. Wait until its window says *Engine running*.

Check it worked. Open a terminal (macOS: **Terminal**; Windows: **PowerShell**; Linux:
your usual one) and type:

```bash
docker info --format 'Server: {{.ServerVersion}}'
```

**Correct output** is a version number:

```
Server: 29.1.3
```

If instead you see `Cannot connect to the Docker daemon`, Docker Desktop is not
running. Open it and wait for *Engine running*.

---

## 2. Install Git LFS

The simulation output is about **2.1 GB**, which is too large for ordinary Git. The
repository stores it with an add-on called Git LFS, and without that add-on you will
download 2.1 GB of unusable placeholder files instead of the data.

```bash
git lfs install
```

**Correct output:**

```
Git LFS initialized.
```

If it says `git: 'lfs' is not a git command`, install it first: macOS
`brew install git-lfs`, Windows `winget install GitHub.GitLFS`, Debian/Ubuntu
`sudo apt install git-lfs`. Then run `git lfs install` again.

---

## 3. Get the code

Pick a folder you can find again, and clone into it:

```bash
cd ~/Documents
git clone https://github.com/ethandecamp/butcher_lab_fsg_exploration.git
cd butcher_lab_fsg_exploration
```

**Correct output** ends with something like `Resolving deltas: 100% ... done.`

---

## 4. Download the simulation data

Cloning does not necessarily bring the large files with it. Fetch them explicitly:

```bash
git lfs pull
```

This downloads roughly 2.1 GB and can take several minutes.

**Check it worked.** Do not use the folder's size for this. `one_way_fsg_model`
also holds several hundred megabytes of ordinary PDFs and images that arrive with any
clone, so a clone that fetched *none* of the simulation data still measures 981 MB --
big enough to look right, and wrong. Check one simulation file instead:

```bash
ls -l "one_way_fsg_model/FSG Results/flow_U0p0540/step_014/solid_fields.h5"
```

**Correct output** shows a size in the millions:

```
-rw-r--r--  1 you  staff  4286796 ... solid_fields.h5
```

If the size is around **132** instead, you have a placeholder, not the data. Run
`git lfs install`, then `git lfs pull` again. (For reference, the whole folder is
`2.1G` once the data is really there, and 981M when it is not.)

---

## 5. Build the image

An *image* is the packaged program. Building it takes a few minutes the first time and
seconds after that.

```bash
cd llm_insights
docker build -t llm-insights:latest .
```

**Correct output** ends with these two lines:

```
 => => naming to docker.io/library/llm-insights:latest
 => => unpacking to docker.io/library/llm-insights:latest
```

**The whole test suite runs during this build.** That is on purpose: if any test fails,
the build fails and no image is produced, so a broken image cannot reach you. You will
see this line go by:

```
430 passed, 29 skipped, 1 warning in 7.37s
```

The 29 skipped ones are the tests that need the simulation data, which is not inside
the image — see the next step for why. The one warning is pytest noticing a class named
`TestSpec` that is not a test class; it is harmless.

### Building for a different kind of computer

Apple Silicon Macs are `arm64`; most desktops and servers are `amd64`. To build one
image that runs on both:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t llm-insights:latest .
```

You only need this if you are sending the image to someone else.

---

## 6. Run it — no internet, no API key

The simulation data is **not** inside the image. It is 2.1 GB, it is Dan's, and baking
a copy of it into a container would freeze one snapshot of his results somewhere nobody
would ever think to look for it. Instead you *mount* the folder: you hand the container
a read-only window onto the copy already on your disk.

```bash
mkdir -p out
docker run --rm \
  --network none \
  -v "$(cd .. && pwd)/one_way_fsg_model:/data/one_way_fsg_model:ro" \
  -v "$(pwd)/out:/out" \
  llm-insights:latest smoke
```

On **Windows PowerShell**, use this instead:

```powershell
mkdir out -Force
docker run --rm `
  --network none `
  -v "${PWD}\..\one_way_fsg_model:/data/one_way_fsg_model:ro" `
  -v "${PWD}\out:/out" `
  llm-insights:latest smoke
```

Reading that command:

| Part | What it does |
|---|---|
| `--rm` | Delete the container when it finishes. Nothing accumulates. |
| `--network none` | **Cut the network entirely.** This run cannot reach the internet even if it wanted to. It is the proof that no model was called. |
| `-v ...one_way_fsg_model:/data/one_way_fsg_model:ro` | Show the container the data folder. The `:ro` on the end means **read-only**: the container physically cannot modify Dan's results. |
| `-v ...out:/out` | Where the report is written on *your* disk. |
| `smoke` | Which mode to run. This one replays recorded claims and needs no model. |

**Correct output** ends like this:

```
  proposed        7
  admissible      7
  rejected        0
  survived        5
  falsified       0
  narrowed        1
  could not run   1

cards:       /out/cards.json
markdown:    /out/report.md
html:        /out/report.html
transcript:  /out/transcript.json
```

Then open the report. It is in the `out` folder you just created:

```bash
open out/report.html        # macOS
xdg-open out/report.html    # Linux
start out\report.html       # Windows
```

---

## 7. The other modes

`smoke` is one of four. Replace the last word of the `docker run` command:

| Mode | What it does |
|---|---|
| `smoke` | Replay the recorded run. No network, no model, no key. **Start here.** |
| `test` | Run the test suite inside the container. |
| `run --question "..."` | Ask a new question. This one **does** need a model — drop `--network none` and see §7a. |
| `shell` | Open a shell inside the container, for poking around. |

### 7a. Asking a new question

`run` needs a language model, and the container has none inside it. The keyless option
is a model running on your own machine (see `INSTRUCTIONS.md` §5b). The container
reaches it through the special hostname `host.docker.internal`:

```bash
docker run --rm \
  -e LLM_INSIGHTS_LOCAL_URL=http://host.docker.internal:11434 \
  -e LLM_INSIGHTS_LOCAL_MODEL=llama3.1:8b \
  -v "$(cd .. && pwd)/one_way_fsg_model:/data/one_way_fsg_model:ro" \
  -v "$(pwd)/out:/out" \
  llm-insights:latest run --generator local --question "Does inlet flow rate change where stress peaks along the cushion?"
```

Note there is no `--network none` here — it would cut off the model as well.
`ANTHROPIC_API_KEY` is always removed inside the container, so a key you happen to have
exported can never turn a keyless run into a billed one.

---

## 8. When something goes wrong

| Symptom | What it means | What to do |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop is not running. | Open Docker Desktop, wait for *Engine running*, retry. |
| `error: no FSG dataset at /data/one_way_fsg_model` | The mount is wrong, or you are pointing at the wrong folder. | Check the path before the `:` in `-v` is the folder that contains `FSG Results`. |
| `error: the dataset at ... is Git LFS pointer files, not the real data` | The folder and filenames are all there, but the files are 132-byte placeholders. | `git lfs install`, then `git lfs pull` (steps 2 and 4). Then rerun. |
| `ValueError: This file contains pickled (object) data` | An older image without the pointer check, hitting the same placeholder problem as the row above. | Rebuild the image (step 5), then follow the row above. |
| The build stops on a red `FAILED` with test output | A test failed, so the image was **not** built. This is the safety net working. | Send the last 40 lines of the build output to Ethan. Do not try to skip the tests. |
| `docker: invalid reference format` | Usually a space in your folder path, or Windows backslashes on macOS/Linux. | Keep the quotes around the `-v` values exactly as written above. |
| `permission denied` writing to `/out` | The `out` folder does not exist or is not writable. | `mkdir -p out` in the `llm_insights` folder first. |
| A card in the report says `ENVIRONMENT FAULT` | That one test could not run on this setup. It is **not** a result about the biology. | Report it; do not read anything into that card. |
| `no space left on device` | Docker's disk allowance is full. | Docker Desktop → Settings → Resources, raise the disk limit; or `docker image prune` to remove images you no longer need. |
| The run finishes but `out/` is empty | The `-v ...:/out` mount is missing or misspelled. | Copy the command from §6 again, unmodified. |
| `exec format error` | The image was built for a different chip. | Rebuild it on this machine with the plain `docker build` from step 5. |

### If you are working in a git worktree

This is for Ethan and anyone else running out of `lane-a/`, `lane-b/` or `lane-c/`
rather than a normal clone. **In a worktree, `one_way_fsg_model` is a symlink**, and a
bind mount of a symlink is not a bind mount of the directory it points at — Docker will
either mount the link itself or refuse. Mount the real directory in the main checkout:

```bash
-v "/absolute/path/to/butcher_lab_fsg_exploration/one_way_fsg_model:/data/one_way_fsg_model:ro"
```

`readlink one_way_fsg_model` prints the real path if you are not sure. A normal user
who cloned the repository has a real directory and should use the §6 command unchanged.

---

## 9. What is not in this container

- **The FEniCS solver.** This image runs the analysis, not the simulation. Re-running
  Dan's model needs FEniCS and gmsh, which is a separate and harder packaging problem.
- **The simulation data.** Mounted, never baked. See §6.
- **Any language model.** `smoke` needs none. `run` reaches one on your own machine.
