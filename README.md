# HDB Resale Flat Explorer

An interactive Streamlit web app for exploring and visualizing Singapore HDB resale flat data. Empower new home buyers and analysts with data-driven insights, dynamic filtering, and beautiful visualizations.

## Features

- 📊 Interactive data exploration and filtering
- 🗺️ Map visualizations with Folium
- 📈 Dynamic charts with Plotly
- 🏙️ Insights by town, flat type, area, and more
- ⚡ Fast, optimized data loading and caching

## Data Sources

- Main dataset: `datasets/train.csv`

## Project Structure

```
├── app.py                  # Main Streamlit app
├── requirements.yml        # Conda environment and pip dependencies
├── datasets/               # Data files
│   ├── train.csv
│   ├── test.csv
│   └── sample_sub_reg.csv
├── BusinessObjective.md    # Project goals
├── Data_Dictionary.md      # Data schema and definitions
```

## Setup Guide

### 1. Install Conda (if not already installed)

- Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

### 2. Create and Activate the Environment

Open a terminal (Anaconda Prompt or cmd) in the project folder and run:

```cmd
conda env create -f requirements.yml
conda activate hdb-explorer
```

### 3. Run the Streamlit App

```cmd
streamlit run app.py
```

The app will open in your browser. If not, follow the link shown in the terminal.

## Credits

- Developed for NTU Data Science & AI coursework
- Built with Streamlit, pandas, plotly, folium, and streamlit-folium
