import requests
import pandas as pd
from io import StringIO

def fetch_data_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch the document. Status code: {response.status_code}")
        return None
    return response.text

def extract_coordinates_and_chars(html_data):
    # Read HTML with StringIO to fix the deprecation warning
    tables = pd.read_html(StringIO(html_data))
    if not tables:
        print("No table found in the document.")
        return None

    df = tables[0]

    # Show original column names for debugging
    print("Original columns:", df.columns.tolist())

    # Use only the first 3 columns and rename them
    df = df.iloc[:, :3]
    df.columns = ['x', 'char', 'y']

    # Remove the first row if it's a header mistakenly inside the table
    if isinstance(df.iloc[0]['x'], str) and 'x' in df.iloc[0]['x'].lower():
        df = df.drop(index=0)

    # Drop rows with missing values
    df = df.dropna()

    # Convert x and y to integers
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)

    return df

def decode_secret_message(url):
    # Step 1: Fetch HTML data
    html_data = fetch_data_from_url(url)
    if not html_data:
        return

    # Step 2: Extract structured data
    data = extract_coordinates_and_chars(html_data)
    if data is None:
        return

    # Step 3: Find grid dimensions
    max_x = data['x'].max()
    max_y = data['y'].max()

    # Step 4: Initialize empty grid
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Step 5: Populate grid with characters
    for _, row in data.iterrows():
        x, char, y = row
        grid[y][x] = char

    # Step 6: Print the decoded message (from top to bottom visually)
    for row in reversed(grid):  # flip vertically
        print(''.join(row))


# URL to the Google Docs page
url = "https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub"
decode_secret_message(url)
