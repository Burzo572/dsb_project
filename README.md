**Bangladesh Flood Prediction — Exploratory Data Analysis**

**Project Overview**

This project explores environmental and hydrological data to better understand the factors associated with flooding in Bangladesh. The main objective is to investigate whether rainfall, soil moisture, river discharge, and historical environmental conditions are related to flood occurrence within the next 24 hours.

The project focuses initially on Exploratory Data Analysis (EDA) to understand the dataset, assess data quality, identify patterns, and prepare the data for potential machine learning models.

**Dataset Description**

The dataset is a daily, multi-station time series covering 30 river monitoring locations across Bangladesh.

Original dataset size: 839,340 rows and 49 columns

Geographical coverage: 30 river stations across four major river basins

Time coverage: 1950–2026

Data frequency: Daily observations per station

Format: Parquet

Dataset source: Bangladesh Flood Data Set — GitHub

Each row represents a particular station on a particular date. The dataset combines environmental variables, river discharge information, historical flood indicators, and engineered features derived from previous observations.

**Data Sources**

The dataset was compiled from multiple free and public sources, including:

Open-Meteo / ERA5 and ERA5-Land: rainfall and soil moisture data.

Open-Meteo Flood API / GloFAS: river discharge estimates.

Global Flood Monitoring System (GFMS): flood detection information.

NASA MCDWD and Copernicus Global Flood Monitoring: satellite-based flood observations.

Dartmouth Flood Observatory and Global Flood Database: historical flood-event information.

Bangladesh Flood Forecasting and Warning Centre (FFWC): information extracted from flood reports.

These sources provide complementary information, but they differ in temporal coverage, measurement methods, and reliability.

**Main Variables**

-------------------------------------
date

Date of the observation

station_id

Unique monitoring station identifier

station_name

Name of the monitoring station

rainfall_local_mm

Local rainfall in millimetres

rainfall_upstream_mm

Rainfall in the upstream catchment

soil_moisture_local

Local soil moisture indicator

river_discharge_m3s

River discharge in cubic metres per second

flood_byStor

Additional flood indicator

flood_within_24h

Flood label for the next 24 hours

flood_within_48h

Flood label for the next 48 hours

flood_within_72h

Flood label for the next 72 hours

---------------------------------------------------------------
The dataset also contains lagged variables, rolling rainfall summaries, trend indicators, missing-value flags, and other derived environmental features.

**Project Objective**

The main research question is:

Can environmental and hydrological variables help predict whether a flood will occur at a monitoring station within the next 24 hours?

The intended task is binary classification:

0 — no flood is recorded by the dataset's labeling procedure for the prediction horizon.

**Limitations**

The dataset combines information from multiple sources with different levels of coverage and reliability.

Some environmental variables are based on reanalysis or modelled products rather than direct measurements at a physical monitoring station.

Flood labels may be incomplete because flood-detection sources do not observe every event.

A label of 0 does not necessarily prove that no flood occurred.

The original date range does not imply that every variable is available for every date.

Relationships identified during EDA do not establish causation.

Good model performance would not, by itself, establish that the model is suitable for operational flood warnings.

1 — a flood is recorded by the dataset's labeling procedure for the prediction horizon.

These labels should not be interpreted as perfect confirmation that flooding did or did not occur in reality.
