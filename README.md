# 🪐 Mars Habitability Explorer

<div align="center">

**A powerful web application for analyzing Mars habitability data and identifying optimal settlement locations**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[Live Demo](https://make-mars-closer.streamlit.app/) • [Documentation](#documentation) 

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Mars Habitability Explorer is an interactive Streamlit application designed to analyze environmental data from Mars and identify the most habitable locations for potential human settlement. The application processes CSV datasets containing atmospheric, temperature, wind, and other environmental parameters, then calculates habitability scores based on customizable weighting criteria.

### Key Capabilities

- **Intelligent Data Processing**: Automatically validates and processes Mars environmental datasets
- **Customizable Scoring**: Adjust parameter weights to prioritize different habitability factors
- **Interactive Visualization**: Generate detailed maps showing habitability scores across Mars
- **Advanced Filtering**: Apply multiple filters to refine candidate locations
- **Export Functionality**: Download filtered results for further analysis

## ✨ Features

### 📤 Data Management
- **Multi-file Upload**: Upload and process multiple CSV files simultaneously
- **Schema Validation**: Automatic validation ensures data integrity before processing
- **Error Handling**: Clear error messages guide users when data doesn't match expected format
- **Real-time Processing**: Instant feedback on file upload and validation status

### 🎚️ Customizable Analysis
- **Parameter Weighting**: Adjust importance of:
  - Temperature conditions
  - Atmospheric pressure
  - Surface water access
  - Solar energy availability
  - Wind speed patterns
- **Dynamic Normalization**: Weights are automatically normalized for consistent scoring
- **Threshold Filtering**: Set minimum scores for each parameter to filter results

### 🗺️ Visualization
- **Interactive Maps**: Mollweide projection maps showing habitability scores
- **Top Locations Highlighting**: Visual identification of best candidate sites
- **Color-coded Scoring**: Intuitive color mapping for quick assessment
- **Multiple Map Views**: Compare overall scores and best location highlights

### 📊 Data Export
- **Filtered Results**: Export top candidate locations as CSV
- **Comprehensive Data**: Includes all calculated scores and original parameters
- **Easy Integration**: Standard CSV format for use in other tools

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **pip** (Python package manager)
- **Git** (for deployment) ([Download](https://git-scm.com/downloads))

## 📦 Installation

### Local Setup

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/yourusername/mars-habitability-explorer.git
   cd mars-habitability-explorer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run habitability_app.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 🚀 Usage

### Getting Started

1. **Upload Data Files**
   - Click "Browse files" in the sidebar
   - Select one or more CSV files with Mars habitability data
   - Wait for validation confirmation (✅ success message)

2. **Configure Parameters**
   - Adjust sliders to set parameter weights (0.0 to 1.0)
   - Set the top percentage threshold (e.g., show top 1% of locations)
   - Optionally set minimum score filters in the expandable section

3. **Calculate Results**
   - Click "Calculate Best Locations" button
   - View normalized weights table
   - Explore the interactive map
   - Review top candidates in the data table

4. **Export Results**
   - Click "Download filtered results as CSV"
   - Use the exported file for further analysis

### Example Workflow

```
1. Upload CSV file(s) → 2. Set weights → 3. Calculate → 4. Explore maps → 5. Export results
```

## 📄 Data Format

### Required Schema

Your CSV files must contain exactly **86 columns** in the following order:

#### Core Parameters (7 columns)
1. `lattitude` - Latitude coordinate
2. `longitude` - Longitude coordinate
3. `atm pressure` - Atmospheric pressure (Pascals)
4. `atm density` - Atmospheric density
5. `temperture` - Temperature (Kelvin)
6. `zonal wind` - Zonal wind component
7. `meridional wind` - Meridional wind component

#### Extended Variables (79 columns)
8-86. `extvar_0` through `extvar_78` - Additional environmental variables

### File Format Requirements

- **Separator**: Semicolon (`;`)
- **Header Row**: First row is skipped (use `skiprows=[0]`)
- **Encoding**: UTF-8 recommended
- **Data Types**: Numeric values only

### Example File Structure

```csv
header_row_to_skip
-90.0;180.0;610.0;0.02;210.0;5.2;-3.1;...;...;...
-89.5;179.5;615.0;0.021;211.0;5.3;-3.0;...;...;...
...
```

> **Note**: The application validates schema automatically. If your file doesn't match, you'll receive a detailed error message indicating what's missing or incorrect.


## 📁 Project Structure

```
mars-habitability-explorer/
│
├── habitabilty_app.py          # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── (optional)
    ├── .gitignore              # Git ignore rules
    ├── LICENSE                 # License file
    └── data/                   # Sample data files (not included)
```

## 🛠️ Technologies Used

### Core Framework
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Python 3.8+](https://www.python.org/)** - Programming language

### Data Processing
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** - Numerical computing

### Visualization
- **[GeoPandas](https://geopandas.org/)** - Geospatial data processing
- **[Matplotlib](https://matplotlib.org/)** - Plotting and visualization
- **[Shapely](https://shapely.readthedocs.io/)** - Geometric operations
- **[PyProj](https://pyproj4.github.io/pyproj/)** - Cartographic projections

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -r requirements-dev.txt  # If available

# Run tests
pytest  # If test suite exists
```

### Code Style

This project follows PEP 8 style guidelines. Consider using:
- **Black** for code formatting
- **Flake8** for linting
- **mypy** for type checking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Open source community for the excellent libraries
- NASA and space agencies for Mars data inspiration

## 📧 Support

For questions, issues, or feature requests:
- Open an issue on [GitHub Issues](https://github.com/yourusername/mars-habitability-explorer/issues)
- Contact: [yousef.mohamed.rabia@gmail.com]

---

<div align="center">

**Made with ❤️ for Mars exploration**

⭐ Star this repo if you find it useful!

</div>
