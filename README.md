# 🪐 Mars Habitability Explorer

A Streamlit web application for exploring Mars habitability datasets. Upload CSV files with Mars environmental data and analyze the most habitable locations based on customizable parameter weights.

## Features

- 📤 **File Upload**: Upload CSV files with Mars habitability data
- ✅ **Schema Validation**: Automatic validation of uploaded data schema
- 🎚️ **Customizable Weights**: Adjust importance of different habitability factors
- 🗺️ **Interactive Maps**: Visualize habitability scores on Mars maps
- 📊 **Data Analysis**: Filter and explore top candidate locations
- 💾 **Export Results**: Download filtered results as CSV

## Deployment to Streamlit Cloud

### Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in (or create an account)
2. Click the "+" icon in the top right → "New repository"
3. Name your repository (e.g., `mars-habitability-app`)
4. Make it **Public** (required for free Streamlit Cloud)
5. Click "Create repository"

### Step 2: Upload Your Files to GitHub

**Option A: Using GitHub Web Interface**

1. In your new repository, click "uploading an existing file"
2. Upload these files:
   - `habitabilty_app.py` (your main app file)
   - `requirements.txt`
   - `README.md` (this file)
3. Click "Commit changes"

**Option B: Using Git (Command Line)**

```bash
cd /Users/yousefrabia/Desktop
git init
git add habitabilty_app.py requirements.txt README.md
git commit -m "Initial commit: Mars Habitability Explorer app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch (usually `main`)
5. Set the **Main file path** to: `habitabilty_app.py`
6. Click "Deploy!"

### Step 4: Wait for Deployment

- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- The first deployment may take 2-5 minutes
- You'll get a URL like: `https://your-app-name.streamlit.app`

## Local Development

To run the app locally:

```bash
pip install -r requirements.txt
streamlit run habitabilty_app.py
```

## Data Format

Your CSV files should have the following columns:
- `lattitude`, `longitude`
- `atm pressure`, `atm density`
- `temperture`
- `zonal wind`, `meridional wind`
- `extvar_0` through `extvar_78` (79 additional variables)

Total: **86 columns**

Files should be semicolon-separated (`;`) with the first row skipped.

## Notes

- The app requires all dependencies listed in `requirements.txt`
- Map visualization requires `geopandas`, `shapely`, `pyproj`, and `matplotlib`
- Files are processed in memory - no data is stored on the server

