<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png">
    <img src="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png" alt="OptiSurvey logo" width="50%" height="50%">
  </picture>
</p>

<p align="center"><b>OptiSurvey</b>: AI-assisted literature survey generation for optical communication research</p>
<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="license" src="https://img.shields.io/badge/license-apache-green">
  <img alt="domain" src="https://img.shields.io/badge/domain-optical%20communication-orange">
</p>

---

## Overview

OptiSurvey is a domain-focused literature survey system for optical communication research. It helps researchers collect papers, parse PDFs, cluster content, draft survey sections, and export the final survey as PDF or LaTeX PDF.

The web app now includes two post-generation workflows:

- `OptiResearch`: a reader + grounded Q&A workspace for the generated survey.
- `OptiResearch cycles`: a multi-cycle research ideation workflow that turns the generated survey into brainstormed ideas, structured hypotheses, and ranked final candidates.

This repository is built around optical communication use cases such as coherent systems, fiber optics, optical networks, silicon photonics, free-space optics, and related signal-processing pipelines.

## Main Features

- Upload local PDFs or bring in recommended references.
- Parse and structure optical communication papers.
- Generate a survey draft from the uploaded paper set.
- Export the survey as a regular PDF or a LaTeX-based PDF.
- Open the generated survey in `OptiResearch` for reader-side inspection.
- Ask grounded questions against the generated survey text in `OptiResearch`.
- Run `OptiResearch` cycles on top of the current survey.
- Export the final selected hypotheses from the latest OptiResearch cycle as Markdown.

## Project Layout

- `src/demo/templates/demo/index.html`: main frontend UI.
- `src/demo/views.py`: backend endpoints for upload, survey generation, PDF export, and OptiResearch workflows.
- `src/static/data/results/`: generated PDFs.
- `src/static/data/txt/`: generated survey text artifacts.
- `src/static/data/info/`: processed survey markdown artifacts.

## Quick Start

OptiSurvey requires Python `3.10`. Plan for at least `20 GB` of free disk space for dependencies, parsed files, and models.

### 1. Clone the repository

```bash
git clone https://github.com/TechnicolorGUO/OptiSurvey
cd OptiSurvey
```

### 2. Create the environment

```bash
conda create -n optisurvey python=3.10
conda activate optisurvey
python scripts/setup_env.py
```

If you hit model download timeout issues, use the fallback flow below:

```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```

### 3. Configure environment variables

Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_generation_model_here>
EVALUATE_MODEL=<your_evaluation_model_here>
VLM_MODEL=<your_vlm_model_here>
```

At minimum, `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `MODEL` are required for the main generation flow. The additional model fields are used by other parts of the system.

### 4. Start the web app

```bash
python src/manage.py runserver 0.0.0.0:8001
```

Then open:

```text
http://localhost:8001
```

## Frontend Workflow

The standard UI flow is:

1. Upload reference PDFs.
2. Let OptiSurvey parse and process the paper set.
3. Generate the survey draft.
4. Preview and optionally edit the survey content in the browser.
5. Export a regular PDF if needed.
6. Export the final LaTeX PDF.
7. Open `OptiResearch` once the survey is ready.
8. Optionally run `OptiResearch` cycles on top of the generated survey.

## OptiResearch

`OptiResearch` is the post-generation research workspace inside the main UI.

### What it does

- Shows the generated survey PDF inside an embedded reader.
- Uses the generated survey text as context for grounded Q&A.
- Keeps the reader and Q&A tied to the current survey `survey_id` once that survey has been loaded in the page.

### Unlock conditions

The reader and the chat area do not unlock at the same time:

- The PDF reader becomes meaningful when a survey PDF exists.
- The grounded chat requires survey text artifacts in addition to the PDF.
- In normal usage, the most reliable path is to generate the LaTeX PDF for the current survey and then open `OptiResearch`.

### Recommended usage

1. Generate the survey normally.
2. Click the LaTeX export button to produce the final survey PDF.
3. Open `OptiResearch`.
4. Read the PDF in the left pane.
5. Ask grounded questions in the right pane.

### Notes for developers

OptiResearch has developer-only fallback behavior controlled in `src/demo/views.py`:

- `OPTIRESEARCH_DEV_MODE`
- `OPTIRESEARCH_DEV_PDF_RELATIVE_PATH`
- `OPTIRESEARCH_DEV_FALLBACK_TO_LATEST`

These flags are currently hardcoded in the backend, not read from `.env`.

Behavior summary:

- `OPTIRESEARCH_DEV_MODE = True`: allows a developer override PDF path and optional fallback to the latest available LaTeX PDF.
- `OPTIRESEARCH_DEV_MODE = False`: disables the developer override path, but the backend can still fall back to the latest available real survey result if one exists on disk.

If you want `OptiResearch` to stay fully locked until the current survey is generated, that behavior should be enforced in `get_optiresearch_state()` rather than only by toggling `OPTIRESEARCH_DEV_MODE`.

## OptiResearch Cycles

`OptiResearch` also includes a structured ideation workflow that runs after a survey exists.

### Pipeline

Each cycle runs three stages:

1. `Brainstorming Agent`
2. `Hypothesis Agent`
3. `Validation Agent`

### Output of each cycle

Each cycle stores:

- an idea pool
- structured hypothesis cards
- ranked hypotheses
- selected final candidates
- a reviewer summary
- a stop / continue decision

### Multi-cycle behavior

The UI supports repeated cycles with an auto-stop rule. In practice, the workflow usually converges in `2` to `3` cycles, unless you increase the maximum iteration count.

### Export final selected hypotheses

The OptiResearch cycles panel now includes an `Export Final` button.

It exports the `selected_candidates` from the latest cycle as a Markdown file containing:

- survey metadata
- cycle number
- reviewer summary
- stop decision
- selected hypothesis cards
- scores, rank, evidence reasoning, and cited papers

The downloaded file name follows this pattern:

```text
optiresearch_selected_hypotheses_<survey_id>_cycle_<xx>.md
```

## Docker

Make sure the repository is cloned and `.env` is configured before using Docker.

### GPU version

```bash
docker build -t optisurvey .
docker run --gpus all -p 8001:8001 optisurvey
```

### CPU version

For CPU-only deployment, remove the CUDA-specific line in `scripts/additional_scripts.py` before building:

```python
"device-mode": "cuda",
```

Then build and run:

```bash
docker build -t optisurvey-cpu .
docker run -p 8001:8001 optisurvey-cpu
```

After startup, open [http://localhost:8001](http://localhost:8001).

## GPU Setup Notes

If you want GPU acceleration for MinerU and related parsing steps, follow the official MinerU setup guides:

- [Ubuntu 22.04 LTS + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md)

## Direct Survey Generation Without Frontend

If you want to run the survey pipeline directly:

### 1. Enter the pipeline directory

```bash
cd src/demo/survey_generation_pipeline
```

### 2. Copy the root `.env`

```bash
cp ../../../.env .
```

### 3. Run the pipeline

```bash
python main.py
```

This runs the sample survey generation flow and writes output artifacts to the local result directory used by the pipeline.

### 4. Adjust the sample configuration

The pipeline entry point contains a minimal example like this:

```python
if __name__ == "__main__":
    root_path = "."
    pdf_path = "./sample_pdfs"
    survey_title = "Advances in Coherent Optical Communication Systems"
    cluster_standard = "method"
    asg_system = ASG_system(root_path, 'test', pdf_path, survey_title, cluster_standard)
    asg_system.download_pdf()
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()
```

Update:

- `pdf_path` to your own paper directory
- `survey_title` to your target topic
- `cluster_standard` to the clustering strategy you want

## Examples

### Markdown example

https://github.com/user-attachments/assets/fdf48927-ae0f-4040-9595-4a509ea62f08

### LaTeX PDF example

https://github.com/user-attachments/assets/db2b08f5-a328-43e1-9ae9-41c09b54214b

## Why OptiSurvey

Compared with generic literature review tools, OptiSurvey is biased toward optical communication research structure and terminology. It is designed to work with domain-specific concepts such as:

- BER
- OSNR
- coherent detection
- WDM and DWDM
- photonic integration
- free-space optical communication
- optical DSP pipelines

## Contact

For questions, suggestions, or collaboration:

- [guobeichen0228@gmail.com](mailto:guobeichen0228@gmail.com)

## License

[Apache License 2.0](LICENSE)
