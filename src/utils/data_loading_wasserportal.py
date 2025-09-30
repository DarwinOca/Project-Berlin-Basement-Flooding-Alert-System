import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from io import StringIO
from fe.constants import API_URLS
import utm


def get_stations(dest_path):
    urls = API_URLS

    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    station_dfs = {}

    for key, url in urls.items():
        try:
            response = requests.get(url)
            response.encoding = 'ISO-8859-1'  # Handle German umlauts like ä, ö, ü
            html = response.text

            # Wrap HTML string in StringIO to avoid FutureWarning
            tables = pd.read_html(StringIO(html))

            if tables:
                station_dfs[key] = tables[0]
                print(
                    f"{key.title()} stations loaded: {len(tables[0])} entries")
        except Exception as e:
            print(f"Failed to fetch {key} stations: {e}")

    station_dfs['groundwater'].to_csv(dest_path /
                                      'stations_groundwater_raw.csv')
    station_dfs['soil'].to_csv(dest_path / 'stations_soil_raw.csv')
    station_dfs['surface'].to_csv(dest_path / 'stations_surface_raw.csv')


def convert_utm_to_latlon(row):
    # Extract zone number and zone letter
    zone_str = row['Projektion'].replace('UTM ', '')  # e.g., '33N'
    zone_number = int(zone_str[:-1])  # e.g., 33
    zone_letter = zone_str[-1]  # e.g., 'N'

    lat, lon = utm.to_latlon(row['Rechts- wert'], row['Hoch- wert'],
                             zone_number, zone_letter)
    return pd.Series({'lat': lat, 'lon': lon})


def process_stations_groundwater(stations_path, dest_path):
    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)
    station_df = pd.read_csv(stations_path)
    station_df[['lat', 'lon']] = station_df.apply(convert_utm_to_latlon,
                                                  axis=1)
    #Filtering: Only stations that measure groundwater level
    station_df = station_df[station_df['Aus- prägung'].str.contains(
        'GW-Stand')]
    station_df = station_df[[
        'Mess- stellen- nummer', 'lat', 'lon',
        'Gelände- oberkante (GOK) (m ü. NHN)'
    ]]
    station_df = station_df.rename(
        columns={
            'Mess- stellen- nummer': 'ID',
            'Gelände- oberkante (GOK) (m ü. NHN)': 'height'
        })
    station_df = station_df.set_index('ID')
    station_df.to_csv(dest_path / 'stations_groundwater.csv')
    print("Processed stations to "
          f"{dest_path / 'stations_groundwater.csv'}")


def build_data_url(station_id,
                   start_date,
                   end_date,
                   url_params,
                   BASE_URL="https://wasserportal.berlin.de/station.php"):
    """
    Builds the query URL for various types of data from Wasserportal.

    Args:
        station_id: Station ID (string)
        start_date: Start date in DD.MM.YYYY format
        end_date: End date in DD.MM.YYYY format
        url_params: parameters to add to the url

    Returns:
        str: Complete URL for data download
    """

    params = {
        "station": station_id,
        "sdatum": start_date,
        "senddatum": end_date
    }

    query_params = url_params | params
    query = "&".join(f"{k}={v}" for k, v in query_params.items())
    return f"{BASE_URL}?{query}"


def download_data_csv(url, dest_path):
    """
    Downloads data from the given URL and saves it as a CSV file at the specified destination path.
    Handles Wasserportal-specific encoding and error conditions.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    content = response.content.decode(
        "latin1")  # Berlin Wasserportal CSVs use Latin1

    # Check if the response contains PHP errors or HTML instead of CSV
    if content.strip().startswith(
            '<') or 'Notice:' in content or 'Error:' in content:
        raise Exception(
            "Server returned HTML/PHP errors instead of CSV data - station may not exist or have no data for this period"
        )

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_data_dataframe(source_path, dest_path):
    """
    Parses the standard dataframe from Wasserportal to a clean format.
    Works for different data types (groundwater, air temperature, etc.)

    Args:
        df: Raw DataFrame from Wasserportal

    Returns:
        pandas.DataFrame: Clean DataFrame with 'date' and 'value' columns
    """
    df = pd.read_csv(source_path, sep=";")
    # Find the row where the time series data starts (after "Datum" header)
    datum_row = None
    for i, row in df.iterrows():
        if any(str(cell).strip().lower() == "datum" for cell in row):
            datum_row = i
            break

    if datum_row is None:
        raise ValueError("Could not find 'Datum' header in the data")

    # Extract the actual time series data starting after the header row
    data_df = df.iloc[datum_row + 1:].copy()
    data_df.columns = ['date', 'value']

    # Clean the data
    data_df = data_df.dropna().copy()
    data_df['date'] = pd.to_datetime(data_df['date'],
                                     format="%d.%m.%Y",
                                     errors='coerce')
    data_df['value'] = pd.to_numeric(data_df['value'].astype(str).str.replace(
        ",", "."),
                                     errors="coerce")

    # Remove any rows where conversion failed
    data_df = data_df.dropna()
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    data_df.to_csv(dest_path, index=False)


def get_multi_station_data(start_date, end_date, config, num_stations=None):
    """
    Downloads and combines data from multiple stations into a single DataFrame.

    Args:
        start_date: Start date in DD.MM.YYYY format
        end_date: End date in DD.MM.YYYY format
        config: Configuration dictionary for the API (e.g., config_groundwater)
        num_stations: Number of stations to process (default: all stations)

    Returns:
        pandas.DataFrame: Combined DataFrame with date as index and value_stationX columns
    """
    messages = []
    # Convert dates to datetime objects and format as YYYY-MM-DD for filenames
    start_dt = datetime.strptime(start_date, '%d.%m.%Y')
    end_dt = datetime.strptime(end_date, '%d.%m.%Y')
    start_formatted = start_dt.strftime('%Y-%m-%d')
    end_formatted = end_dt.strftime('%Y-%m-%d')

    station_dfs_list = []
    dest_path_extracted = Path('../data/wasserportal/extracted/measurements/')
    dest_path_raw = Path('../data/wasserportal/raw/measurements/')

    # Determine how many stations to process
    station_df = pd.read_csv('../data/wasserportal/stations_groundwater.csv',
                             index_col='ID')
    stations_to_process = station_df.index if num_stations is None else station_df.index[:
                                                                                         num_stations]

    for i, station in enumerate(stations_to_process, 1):
        try:
            # Get the data for one station - use original date format for API
            url_station = build_data_url(station, start_date, end_date, config)
            download_data_csv(
                url_station, dest_path_raw /
                f'station_{station}_{start_formatted}_{end_formatted}.csv')
            parse_data_dataframe(
                dest_path_raw /
                f'station_{station}_{start_formatted}_{end_formatted}.csv',
                dest_path_extracted /
                f'station_{station}_{start_formatted}_{end_formatted}.csv')
            df_station = pd.read_csv(
                dest_path_extracted /
                f'station_{station}_{start_formatted}_{end_formatted}.csv'
            ).rename(columns={'value': f'value_{station}'})
            df_station = df_station.set_index('date')

            station_dfs_list.append(df_station)
        except Exception as e:
            print(f"Failed to process station {station}: {e}")
            continue

        # Print progress every 20 stations
        if i % 20 == 0:
            message = f"Processed {i} stations out of {len(stations_to_process)}"
            print(message)
            messages.append(message)

    if not station_dfs_list:
        raise ValueError("No station data was successfully downloaded")

    # Concatenate all DataFrames in the list into one.
    # axis=1 tells pandas to combine them as new columns.
    # By default, it performs an 'outer' join on the index (our dates),
    # so all dates are kept, with NaN for missing values.
    final_df = pd.concat(station_dfs_list, axis=1)
    dest_path_processed = Path('../data/wasserportal/processed/')
    dest_path_processed.mkdir(parents=True, exist_ok=True)
    file_parquet = f'gw_data_{start_formatted}_{end_formatted}.parquet'
    file_parquet = dest_path_processed / file_parquet
    final_df.to_parquet(file_parquet)

    message = f"DataFrame saved to {file_parquet}"
    print(message)
    messages.append(message)

    return {"file": file_parquet, "messages": messages}


def get_gw_master(start_date, end_date, config, num_stations=None):
    """
    Downloads and combines data from multiple stations into a single DataFrame.
    Stacks the data so the result has columns: date, station, value.

    Args:
        start_date: Start date in DD.MM.YYYY format
        end_date: End date in DD.MM.YYYY format
        config: Configuration dictionary for the API (e.g., config_groundwater)
        num_stations: Number of stations to process (default: all stations)

    Returns:
        dict: {"file": parquet_path, "messages": messages}
    """
    messages = []
    start_dt = datetime.strptime(start_date, '%d.%m.%Y')
    end_dt = datetime.strptime(end_date, '%d.%m.%Y')
    start_formatted = start_dt.strftime('%Y-%m-%d')
    end_formatted = end_dt.strftime('%Y-%m-%d')

    records = []
    dest_path_extracted = Path('../data/wasserportal/extracted/measurements/')
    dest_path_raw = Path('../data/wasserportal/raw/measurements/')

    station_df = pd.read_csv('../data/wasserportal/stations_groundwater.csv',
                             index_col='ID')
    stations_to_process = station_df.index if num_stations is None else station_df.index[:
                                                                                         num_stations]

    for i, station in enumerate(stations_to_process, 1):
        try:
            url_station = build_data_url(station, start_date, end_date, config)
            download_data_csv(
                url_station, dest_path_raw /
                f'station_{station}_{start_formatted}_{end_formatted}.csv')
            parse_data_dataframe(
                dest_path_raw /
                f'station_{station}_{start_formatted}_{end_formatted}.csv',
                dest_path_extracted /
                f'station_{station}_{start_formatted}_{end_formatted}.csv')
            df_station = pd.read_csv(
                dest_path_extracted /
                f'station_{station}_{start_formatted}_{end_formatted}.csv')
            df_station['station'] = station
            records.append(df_station[['date', 'station', 'value']])
        except Exception as e:
            print(f"Failed to process station {station}: {e}")
            continue

        if i % 20 == 0:
            message = f"Processed {i} stations out of {len(stations_to_process)}"
            print(message)
            messages.append(message)

    if not records:
        raise ValueError("No station data was successfully downloaded")

    final_df = pd.concat(records, ignore_index=True)
    dest_path_processed = Path('../data/wasserportal/processed/')
    dest_path_processed.mkdir(parents=True, exist_ok=True)
    file_parquet = f'gw_master_{start_formatted}_{end_formatted}.parquet'
    file_parquet = dest_path_processed / file_parquet
    final_df.to_parquet(file_parquet, index=False)

    message = f"DataFrame saved to {file_parquet}"
    print(message)
    messages.append(message)

    return {"file": file_parquet, "messages": messages}
