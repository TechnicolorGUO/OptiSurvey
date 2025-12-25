<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png">
    <img src="https://github.com/TechnicolorGUO/OptiSurvey/blob/main/resources/logo1.png" alt="Logo" width="50%" height="50%">
  </picture>
</p>

<p align="center"><b>OptiSurvey</b>: The First AI-Powered Literature Survey System Tailored for Optical Communication Research</p>
<p align="center">
  <img alt="python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="Static Badge" src="https://img.shields.io/badge/license-apache-green">
  <img alt="domain" src="https://img.shields.io/badge/domain-optical%20communication-orange">
</p>
<div align="center">
<hr>

[Quick Start](#quick-start) | [Use Docker(Recommended)](#use-docker)

</div>

---

## Introduction

**OptiSurvey** is a specialized **AI-powered literature survey generation system** designed exclusively for **optical communication research**. As optical communication technologies rapidly evolve—spanning fiber optics, free-space optical communications, silicon photonics, coherent detection, quantum communication, and next-generation networking—researchers face the challenge of keeping pace with an exponentially growing body of literature.

OptiSurvey addresses this challenge by leveraging cutting-edge **Large Language Models (LLMs)** and **Natural Language Processing (NLP)** techniques to provide optical communication researchers with an **intelligent**, **interactive**, and **personalized** tool for conducting comprehensive literature reviews.

### 🌟 Why OptiSurvey for Optical Communication?

- **🔬 Domain-Specific Intelligence**: Purpose-built to understand optical communication terminology, concepts, and research paradigms, including modulation formats, channel modeling, signal processing, photonic devices, and network architectures.
- **🎯 Specialized Paper Analysis**: Intelligently extracts and organizes key information relevant to optical communication, such as transmission rates, wavelength ranges, BER performance, system configurations, and experimental setups.
- **📊 Research Trend Identification**: Automatically identifies emerging trends in optical communication research, from WDM systems to space-division multiplexing, AI-enabled optical networks, and beyond.

### 🔥 Key Features:

- **📝 Automatic Literature Survey Generation**: Extract key insights from optical communication papers and generate well-structured, publication-ready literature surveys.
- **💡 Interactive Exploration**: Dynamically filter, refine, and customize your survey in real-time based on specific topics (e.g., coherent detection, ROADM, or optical amplifiers).
- **🔍 Intelligent Clustering**: Automatically categorize papers by research methods, technologies, or application domains specific to optical communication.
- **📄 Professional PDF Export**: Generate high-quality literature surveys in PDF format with either **Markdown** or **LaTeX**, formatted to academic standards.
- **⚡ Multimodality Support**: Extract and analyze figures, diagrams, and experimental setup illustrations from optical communication papers.
- **🐳 Docker Support**: Quickly deploy and run the application in a containerized environment for seamless collaboration.  


![flochart](/resources/flowchart.png)

---

## 🔭 Application Scenarios in Optical Communication

**OptiSurvey** is specifically designed to support optical communication researchers across various domains:

### Research Areas Covered:
- **📡 Fiber Optic Communication**: High-speed transmission systems, optical amplifiers (EDFA, Raman, SOA), dispersion compensation
- **🌐 Optical Networks**: ROADM, WDM/DWDM systems, software-defined optical networks (SDON), elastic optical networks
- **💫 Free-Space Optical Communications**: FSO systems, atmospheric turbulence mitigation, hybrid RF/FSO networks
- **🔬 Silicon Photonics**: Photonic integrated circuits, on-chip optical interconnects, modulators and detectors
- **🎯 Advanced Modulation & Detection**: Coherent detection, QAM, OFDM, probabilistic shaping, DSP algorithms
- **🔐 Quantum Communication**: QKD systems, quantum networks, entanglement distribution
- **📶 Next-Gen Technologies**: Space-division multiplexing, orbital angular momentum (OAM), AI-enabled optical networks

### Use Cases:
✅ **PhD Students**: Quickly review state-of-the-art before starting research  
✅ **Researchers**: Stay updated with latest advances in specific optical communication topics  
✅ **Lab Groups**: Organize and synthesize team's collected papers into coherent surveys  
✅ **Grant Writers**: Generate comprehensive background sections for research proposals  
✅ **Course Instructors**: Create up-to-date teaching materials on optical communication topics

<hr>

## Examples

See how **OptiSurvey** automatically generates comprehensive literature surveys for optical communication research. The system intelligently organizes papers, extracts key technical details, and produces publication-ready documents.

### Markdown Example
https://github.com/user-attachments/assets/fdf48927-ae0f-4040-9595-4a509ea62f08

### LaTeX Example


https://github.com/user-attachments/assets/db2b08f5-a328-43e1-9ae9-41c09b54214b


<hr>

## Quick Start

OptiSurvey requires Python 3.10. A minimum 20G disk space is required for the system and models.

### 1️⃣ Clone the Repository  
Clone the repository to your local machine:  
```sh
git clone https://github.com/TechnicolorGUO/OptiSurvey
cd OptiSurvey
```

### 2️⃣ Set Up the Environment
Create a virtual environment and activate it:
```sh
conda create -n optisurvey python=3.10
conda activate optisurvey
```
Install the required dependencies:
```sh
python scripts/setup_env.py
```

For the `ConnectTimeout` error when downloading Huggingface models, please run the following script:
```bash
pip install modelscope
wget https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/scripts/download_models.py -O download_models.py
python download_models.py
```

### 3️⃣ Configure Environment Variables
Create a  `.env` file in the root directory of the project and add the following configurations:
```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_preferred_model_here>
```
Replace the placeholders with your actual OpenAI API key, API base URL, and preferred model.

To test the system on GPU, you also need to follow the instructions provided by [MinerU](https://github.com/opendatalab/MinerU/tree/master):
- [Ubuntu 22.04 LTS + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](https://github.com/opendatalab/MinerU/blob/master/docs/README_Windows_CUDA_Acceleration_en_US.md)

### 4️⃣ Run the Application
Start the development server by running the following command:
```sh
python src/manage.py runserver 0.0.0.0:8001
```
_(Replace 8001 with any available port number of your choice.)_

### 5️⃣ Access the Application
Once the server is running, open your browser and navigate to:
```
http://localhost:8001
```
You can now use **OptiSurvey** to upload optical communication papers, analyze research trends, and generate comprehensive literature surveys tailored to your research focus!

<hr>

## Use Docker
Before proceeding, ensure you have cloned the repository and configured your `.env` file in the root directory of the project. The `.env` file must include the following configurations:
```env
OPENAI_API_KEY=<your_openai_api_key_here>
OPENAI_API_BASE=<your_openai_api_base_here>
MODEL=<your_preferred_model_here>
```
Replace the placeholders with your actual OpenAI API key, API base URL, and preferred model.

### GPU Version
If you have GPU support, you can build and run the GPU version of the Docker container using the following commands:
```bash
# Build the Docker image
docker build -t optisurvey .

# Run the Docker container (with GPU support)
docker run --gpus all -p 8001:8001 optisurvey
```

### CPU Version
If you do not have GPU support, you can run the CPU version of the Docker container. *​Note*: Before building and running, you need to manually remove the following line from the `scripts/additional_scripts.py` file:
```python
"device-mode": "cuda",
```
Then run the following commands:
```bash
# Build the Docker image
docker build -t optisurvey-cpu .

# Run the Docker container (with CPU support)
docker run -p 8001:8001 optisurvey-cpu
```

After starting the container, access (http://localhost:8001)[http://localhost:8001] to confirm that the application is running correctly.

<hr>

## Direct Survey Generation Without Frontend

If you want to generate surveys directly without using the frontend, follow these steps:

1. Navigate to the `src/demo/survey_generation_pipeline` directory:
```bash
cd src/demo/survey_generation_pipeline
```
2. Copy the `.env` file to this directory. If you already have a .env file in the root of your project, you can copy it like this:
```bash
cp ../../../.env .
```
*Note*: Ensure the `.env` file contains the required configurations (e.g., `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `MODEL`).

3. Run the pipeline directly:
```bash
python main.py
```
This will execute the survey generation pipeline on our sample PDFs and output the results (.md and .pdf) to the `result` folder directly.

4. Modify the script for your own sample
The `main.py` contains the following code to generate a survey:

```python
if __name__ == "__main__":
    root_path = "."
    pdf_path = "./sample_pdfs" #Set this to the path of the folder containing your PDF files.
    survey_title = "Advances in Coherent Optical Communication Systems" #Set this to the title of your survey.
    cluster_standard = "method" #Set this to the clustering standard you want to use (e.g., method, technology, application).
    asg_system = ASG_system(root_path, 'test', pdf_path, survey_title, cluster_standard) #test refers to the survey_id which prevents re-parsing pdfs.
    asg_system.download_pdf() 
    # Downloads PDFs to "./sample_pdfs". Add your own optical communication papers for analysis.
    asg_system.parsing_pdfs()
    asg_system.description_generation()
    asg_system.agglomerative_clustering()
    asg_system.outline_generation()
    asg_system.section_generation()
    asg_system.citation_generation()
```

---

## 🚀 Why Choose OptiSurvey?

Unlike generic literature review tools, **OptiSurvey** understands the unique characteristics of optical communication research:
- Recognizes domain-specific terminology and metrics (BER, OSNR, dispersion, spectral efficiency, etc.)
- Identifies relationships between different optical technologies and methods
- Understands the evolution of optical communication standards and technologies
- Provides intelligent clustering tailored to optical communication research paradigms

## 📬 Contact

For questions, suggestions, or collaborations related to **OptiSurvey**, please contact:
📧 [guobeichen0228@gmail.com](mailto:guobeichen0228@gmail.com)

We welcome feedback from the optical communication research community!

## 📄 License

[Apache License 2.0](LICENSE)
