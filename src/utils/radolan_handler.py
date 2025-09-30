import re
import os
from datetime import datetime


class RadolanFileHandler:
    """
    Handler for RADOLAN files with date-based filtering capabilities.
    """

    def __init__(self, data_directory=None):
        """
        Initialize the RadolanFileHandler.
        
        Parameters:
        -----------
        data_directory : str, optional
            Base path to directory containing monthly RADOLAN folders
        """
        self.data_directory = data_directory
        self.filename_pattern = re.compile(
            r'raa01-sf_10000-(\d{10})-dwd---bin')
        self.archive_pattern = re.compile(r'SF(\d{6})\.tar\.gz')

    def parse_timestamp(self, timestamp_str):
        """
        Parse 10-digit timestamp from RADOLAN filename.
        
        Parameters:
        -----------
        timestamp_str : str
            10-digit timestamp string (YYMMDDHHMM)
            
        Returns:
        --------
        datetime
            Parsed datetime object (time set to 00:00 for daily filtering)
        """
        yy = int(timestamp_str[:2])
        mm = int(timestamp_str[2:4])
        dd = int(timestamp_str[4:6])
        hh = int(timestamp_str[6:8])
        min_val = int(
            timestamp_str[8:10])  # Ignore hh and mm for daily filtering

        # Convert 2-digit year to 4-digit (assuming 2000-2099 range)
        year = 2000 + yy

        # Return date with time set to 00:00 for daily comparison
        return datetime(year, mm, dd, hh, min_val)

    def extract_date_from_filename(self, filename):
        """
        Extract datetime from RADOLAN filename.
        
        Parameters:
        -----------
        filename : str
            RADOLAN filename
            
        Returns:
        --------
        datetime
            Extracted datetime or None if parsing fails
        """
        match = self.filename_pattern.search(filename)
        if match:
            timestamp_str = match.group(1)
            return self.parse_timestamp(timestamp_str)
        return None

    def get_monthly_folder_path(self, year, month, data_directory=None):
        """
        Get the path to monthly folder containing RADOLAN files.
        
        Parameters:
        -----------
        year : int
            Year (4-digit)
        month : int
            Month (1-12)
        data_directory : str, optional
            Base directory (overrides instance directory)
            
        Returns:
        --------
        str
            Path to monthly folder (YYYYMM format)
        """
        directory = data_directory or self.data_directory
        if not directory:
            return None

        # Format: YYYYMM
        monthly_folder = f"{year:04d}{month:02d}"
        return os.path.join(directory, monthly_folder)

    def get_filenames_in_date_range(self,
                                    start_date,
                                    end_date,
                                    data_directory=None):
        """
        Get RADOLAN filenames within specified date range.
        Searches through monthly folders and filters by date only (ignoring time).
        
        Parameters:
        -----------
        start_date : str or datetime
            Start date (format: 'YYYY-MM-DD' or datetime object)
        end_date : str or datetime
            End date (format: 'YYYY-MM-DD' or datetime object)
        data_directory : str, optional
            Base directory to search (overrides instance directory)
            
        Returns:
        --------
        list
            List of full file paths within date range, sorted by timestamp
        """
        directory = data_directory or self.data_directory
        if not directory:
            print("No data directory specified")
            return []

        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            start_date = start_date.date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            end_date = end_date.date()

        matching_files = []

        # Generate list of months to search
        current_date = datetime(start_date.year, start_date.month, 1)
        end_month = datetime(end_date.year, end_date.month, 1)

        while current_date <= end_month:
            monthly_folder = self.get_monthly_folder_path(
                current_date.year, current_date.month, directory)

            if os.path.exists(monthly_folder):
                # Scan files in monthly folder
                for filename in os.listdir(monthly_folder):
                    file_date = self.extract_date_from_filename(filename)
                    if file_date and start_date <= file_date.date(
                    ) <= end_date:
                        full_path = os.path.join(monthly_folder, filename)
                        matching_files.append((full_path, file_date))
            else:
                print(f"Monthly folder not found: {monthly_folder}")

            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1,
                                                    month=1)
            else:
                current_date = current_date.replace(month=current_date.month +
                                                    1)

        # Sort by date and return full paths only
        matching_files.sort(key=lambda x: x[1])
        return [filepath for filepath, _ in matching_files]

    def load_radolan_data(self, filepath):
        """
        Load RADOLAN data from binary file.
        
        Parameters:
        -----------
        filepath : str
            Full path to RADOLAN file
            
        Returns:
        --------
        np.array
            RADOLAN precipitation data (900x900)
        """
        import wradlib as wrl

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"RADOLAN file not found: {filepath}")

        # Load RADOLAN data using wradlib
        data, _ = wrl.io.read_radolan_composite(filepath)
        return data

    @staticmethod
    def generate_filename(dt: datetime):  # , file_type='recent'):
        """
        Generate RADOLAN filename from datetime.
        
        Args:
            dt: datetime object
            # file_type: 'recent' or 'historical'
            
        Returns:
            Generated filename
        """
        # if file_type == 'recent':
        timestamp = f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}"
        return f"raa01-sf_10000-{timestamp}-dwd---bin.gz"
        # elif file_type == 'historical':
        #     return f"SF{dt.year}{dt.month:02d}.tar.gz"
        # else:
        #     raise ValueError(f"Unknown file_type: {file_type}")
