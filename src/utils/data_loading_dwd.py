from radolan_handler import RadolanFileHandler

file_handler = RadolanFileHandler()


def generate_radolan_urls_historical(
    start_year,
    end_year,
    start_month=1,
    end_month=12,
    return_filenames=False,
    base_url="https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/radolan/historical/bin/"
):
    """
    Generate RADOLAN filename patterns like SF202112.tar.gz
    
    Args:
        start_year (int): Starting year (must be > 2000)
        end_year (int): Ending year (inclusive, must be > 2000)
        start_month (int): Starting month (1-12), default=1
        end_month (int): Ending month (1-12), default=12
    
    Returns:
        urls: List of URL strings
        
    Raises:
        ValueError: If inputs are invalid
    """
    from datetime import datetime
    # Input validation
    if not all(
            isinstance(x, int)
            for x in [start_year, end_year, start_month, end_month]):
        raise ValueError("All parameters must be integers")

    if start_year <= 2000 or end_year <= 2000:
        raise ValueError("Years must be greater than 2000")

    if not (1 <= start_month <= 12) or not (1 <= end_month <= 12):
        raise ValueError("Months must be between 1 and 12")

    if start_year > end_year:
        raise ValueError("Start year cannot be greater than end year")

    urls = []
    filenames = []

    for year in range(start_year, end_year + 1):
        # For first year, start from start_month
        # For last year, end at end_month
        # For middle years, use full range 1-12

        if year == start_year and year == end_year:
            # Same year case
            month_range = range(start_month, end_month + 1)
        elif year == start_year:
            # First year
            month_range = range(start_month, 13)
        elif year == end_year:
            # Last year
            month_range = range(1, end_month + 1)
        else:
            # Middle years
            month_range = range(1, 13)

        for month in month_range:
            # Generate filename directly for historical files
            filename = f"SF{year}{month:02d}.tar.gz"
            # filename = file_handler.generate_filename(datetime(year, month, 1))
            url = f"{base_url}{year}/{filename}"
            urls.append(url)
            filenames.append(filename)

    if return_filenames:
        return urls, filenames
    return urls


def generate_radolan_urls_recent(
        start_date,
        end_date,
        base_url="https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/radolan/recent/bin/",
        hour=12,
        minute=50,
        return_filenames=False):
    """
    Generate recent RADOLAN URLs for one file per day.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        hour (int): Hour of day to select (0-23), default=12 (noon)
        minute (int): Minute to select (default=50, matching DWD pattern)
        return_filenames (bool): If True, return (urls, filenames) tuple
    
    Returns:
        list or tuple: List of URL strings, or (urls, filenames) tuple if return_filenames=True
    """
    from datetime import datetime, timedelta

    # Parse dates and validation (same as before)
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")

    if not (0 <= hour <= 23):
        raise ValueError("Hour must be between 0 and 23")
    if not (0 <= minute <= 59):
        raise ValueError("Minute must be between 0 and 59")

    urls = []
    filenames = []
    current_date = start_dt

    while current_date <= end_dt:
        dt = current_date.replace(hour=hour, minute=minute)
        # print(dt)

        filename = file_handler.generate_filename(dt)

        # timestamp = f"{dt.year % 100:02d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}"
        # filename = f"raa01-sf_10000-{timestamp}-dwd---bin.gz"
        url = f"{base_url}{filename}"

        urls.append(url)
        filenames.append(filename)
        current_date += timedelta(days=1)

    if return_filenames:
        return urls, filenames
    return urls


def download_file(url, dest_path, timeout=30):
    """
    Download a single file from URL to destination path.
    
    Args:
        url (str): URL to download from
        dest_path (Path or str): Destination file path
        timeout (int): Request timeout in seconds
    
    Returns:
        dict: Result with 'success' (bool), 'message' (str), 'filename' (str)
    """
    import requests
    from pathlib import Path

    dest_path = Path(dest_path)
    filename = dest_path.name

    # Skip if file already exists
    if dest_path.exists():
        return {
            'success': True,
            'message': 'Already exists',
            'filename': filename
        }

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.ok:
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return {
                'success': True,
                'message': 'Downloaded successfully',
                'filename': filename
            }
        else:
            return {
                'success': False,
                'message': f'HTTP {response.status_code}',
                'filename': filename
            }
    except Exception as e:
        return {'success': False, 'message': str(e), 'filename': filename}


def extract_radolan_recent(gz_path, extract_to):
    """
    Extract recent RADOLAN .gz file (single binary file) into YYMM subdirectory.
    
    Args:
        gz_path (str/Path): Path to raa01-sf_10000-*.gz file
        extract_to (str/Path): Base directory to extract file to
    """
    import tarfile
    import gzip
    import shutil
    from pathlib import Path
    import re

    gz_path = Path(gz_path)
    extract_path = Path(extract_to)

    # Use RadolanFileHandler to extract date information
    filename = gz_path.name
    # Remove .gz extension for parsing
    base_filename = gz_path.stem

    file_dt = file_handler.extract_date_from_filename(base_filename)

    if file_dt:
        # Format as YYYYMM for directory name
        yymm = f"{file_dt.year}{file_dt.month:02d}"
        yymm_path = extract_path / yymm
        yymm_path.mkdir(parents=True, exist_ok=True)
    else:
        # Fallback to original path if pattern doesn't match
        yymm_path = extract_path
        yymm_path.mkdir(parents=True, exist_ok=True)
        yymm = "root"

    # Remove .gz extension for output filename
    output_filename = gz_path.stem  # removes .gz
    output_path = yymm_path / output_filename

    print(f"Extracting {gz_path.name} to {yymm}/{output_filename}")

    with gzip.open(gz_path, 'rb') as gz_file:
        with open(output_path, 'wb') as output_file:
            shutil.copyfileobj(gz_file, output_file)

    return output_path


def download_radolan_recent(start_date,
                            end_date,
                            dest_folder,
                            hour=12,
                            minute=50):
    """
    Download recent RADOLAN files for a date range into YYMM subdirectories.
    """
    import re
    from pathlib import Path

    # Generate URLs and filenames
    urls, filenames = generate_radolan_urls_recent(start_date,
                                                   end_date,
                                                   hour=hour,
                                                   minute=minute,
                                                   return_filenames=True)

    dest_path = Path(dest_folder)
    results = {'success': 0, 'failed': 0, 'total': len(urls)}
    failed_downloads = []

    print(f"Starting download of {len(urls)} RADOLAN files...")

    for i, (url, filename) in enumerate(zip(urls, filenames), 1):
        # Remove .gz extension for parsing
        base_filename = filename.replace(
            '.gz', '') if filename.endswith('.gz') else filename
        file_dt = file_handler.extract_date_from_filename(base_filename)

        if file_dt:
            # Format as YYYYMM for directory name
            yymm = f"{file_dt.year}{file_dt.month:02d}"
            yymm_path = dest_path / yymm
            yymm_path.mkdir(parents=True, exist_ok=True)
            file_path = yymm_path / filename
        else:
            # Fallback to original path if pattern doesn't match
            dest_path.mkdir(parents=True, exist_ok=True)
            file_path = dest_path / filename
            yymm = "root"

        # Download single file
        result = download_file(url, file_path)

        if result['success']:
            print(
                f"[{i}/{len(urls)}] {result['message']}: {result['filename']} -> {yymm}/"
            )
            results['success'] += 1
        else:
            print(
                f"[{i}/{len(urls)}] Failed: {result['filename']} - {result['message']}"
            )
            results['failed'] += 1
            failed_downloads.append({'url': url, 'error': result['message']})

    # Summary
    print(f"\nDownload Summary:")
    print(f"  Total files: {results['total']}")
    print(f"  Successful: {results['success']}")
    print(f"  Failed: {results['failed']}")

    return {**results, 'failed_downloads': failed_downloads}


def import_radolan_recent(start_date,
                          end_date,
                          dest_folder,
                          hour=12,
                          minute=50,
                          keep_raw=True):
    """
    Download and extract recent RADOLAN files for a date range into YYMM subdirectories.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        dest_folder (str or Path): Base destination folder
        hour (int): Hour of day to select (0-23), default=12 (noon)
        minute (int): Minute to select (default=50, matching DWD pattern)
        keep_raw (bool): Whether to keep downloaded .gz files, default=True
    
    Returns:
        dict: Results with counts and lists of extracted files
    """
    from pathlib import Path

    dest_path = Path(dest_folder)
    raw_path = dest_path / "raw" / "recent"
    extracted_path = dest_path / "extracted"

    # Step 1: Download files (now organizes into YYMM subdirs)
    print("=== DOWNLOADING FILES ===")
    download_results = download_radolan_recent(start_date,
                                               end_date,
                                               raw_path,
                                               hour=hour,
                                               minute=minute)

    if download_results['success'] == 0:
        print("No files downloaded successfully. Skipping extraction.")
        return download_results

    # Step 2: Extract downloaded files (now preserves YYMM structure)
    print("\n=== EXTRACTING FILES ===")
    extracted_files = []
    extraction_failed = []
    processed_yymm_dirs = set()

    # Get list of downloaded .gz files from all YYMM subdirectories
    gz_files = list(raw_path.rglob("*.gz"))

    for gz_file in gz_files:
        try:
            extracted_file = extract_radolan_recent(gz_file, extracted_path)
            extracted_files.append(extracted_file)

            # Track which YYMM directory this file came from
            yymm_dir = gz_file.parent
            processed_yymm_dirs.add(yymm_dir)

            # Optionally delete raw file
            if not keep_raw:
                gz_file.unlink()
                print(f"  Deleted raw file: {gz_file.name}")

        except Exception as e:
            print(f"  Failed to extract {gz_file.name}: {e}")
            extraction_failed.append(str(gz_file))

    cleaned_dirs = 0
    if not keep_raw:
        print("\n=== CLEANING UP EMPTY DIRECTORIES ===")
        for yymm_dir in processed_yymm_dirs:
            try:
                # Check if directory is empty
                if yymm_dir.exists() and not any(yymm_dir.iterdir()):
                    yymm_dir.rmdir()
                    print(f"  Deleted empty directory: {yymm_dir.name}")
                    cleaned_dirs += 1
            except OSError as e:
                print(f"  Could not delete directory {yymm_dir.name}: {e}")

    # Summary
    print(f"\n=== EXTRACTION SUMMARY ===")
    print(f"  Files extracted: {len(extracted_files)}")
    print(f"  Extraction failed: {len(extraction_failed)}")
    print(f"  Raw files kept: {len(gz_files) if keep_raw else 0}")
    if not keep_raw:
        print(f"  Empty directories cleaned: {cleaned_dirs}")

    return {
        **download_results, 'extracted_files': extracted_files,
        'extraction_failed': extraction_failed,
        'raw_files_kept': keep_raw,
        'directories_cleaned': cleaned_dirs if not keep_raw else 0
    }


def should_keep_file(filename, start_dt, end_dt, time_to_keep):
    """
    Check if file should be kept based on date range and timestamp criteria.
    
    Args:
        filename (str): Filename to check (e.g., 'raa01-sf_10000-0707121250-dwd---bin')
        start_dt (datetime): Start date for filtering
        end_dt (datetime): End date for filtering  
        time_to_keep (int): Desired timestamp (HHMM format, e.g., 1250 for 12:50)
    
    Returns:
        bool: True if file should be kept, False otherwise
    """
    # Use RadolanFileHandler to extract date information
    file_dt = file_handler.extract_date_from_filename(filename)

    if not file_dt:
        return False

    # Check if date is within range
    if not (start_dt <= file_dt <= end_dt):
        return False

    # Extract time components and check if timestamp matches desired time
    file_time = file_dt.hour * 100 + file_dt.minute  # Convert to HHMM format
    if file_time != time_to_keep:
        return False

    return True


def extract_selective_files_from_archive(archive_file, extracted_path,
                                         start_dt, end_dt, time_to_keep):
    """Extract only files matching date range and timestamp from archive."""
    import tarfile
    import re

    extracted_count = 0
    skipped_count = 0

    # Extract YYMM from archive filename (e.g., SF202102.tar.gz -> 202102)
    archive_name = archive_file.name
    yymm_match = re.search(r'SF(\d{6})\.tar\.gz', archive_name)
    if yymm_match:
        yymm = yymm_match.group(1)  # Extract YYYYMM
        yymm_path = extracted_path / yymm
        yymm_path.mkdir(exist_ok=True)
    else:
        # Fallback to original path if pattern doesn't match
        yymm_path = extracted_path
        yymm = "root"

    try:
        with tarfile.open(archive_file, 'r:gz') as tar:
            members = tar.getmembers()

            for member in members:
                if should_keep_file(member.name, start_dt, end_dt,
                                    time_to_keep):
                    extracted_file_path = yymm_path / member.name

                    if extracted_file_path.exists():
                        skipped_count += 1
                    else:
                        tar.extract(member, path=yymm_path)
                        print(f"  Extracted: {member.name} -> {yymm}/")
                        extracted_count += 1

        return True, extracted_count, skipped_count, None

    except Exception as e:
        return False, 0, 0, str(e)


def import_radolan_historical(start_date,
                              end_date,
                              dest_folder,
                              time_to_keep=1250,
                              keep_raw=False):
    """
    Download historical RADOLAN archives, extract only needed files, optionally delete archives.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        dest_folder (str): Base destination folder
        time_to_keep (int): Keep only files with this timestamp (HHMM format, e.g., 1250 for 12:50)
        keep_raw (bool): Whether to keep the original tar.gz files
    """
    import tarfile
    from pathlib import Path
    from datetime import datetime, timedelta

    # Parse dates and validation
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD': {e}")
    if start_dt > end_dt:
        raise ValueError(
            f"Start date ({start_date}) cannot be after end date ({end_date})")

    urls, filenames = generate_radolan_urls_historical(start_dt.year,
                                                       end_dt.year,
                                                       start_dt.month,
                                                       end_dt.month,
                                                       return_filenames=True)

    dest_path = Path(dest_folder)
    archive_path = dest_path / "raw" / "historical"
    extracted_path = dest_path / "extracted"

    archive_path.mkdir(parents=True, exist_ok=True)
    extracted_path.mkdir(parents=True, exist_ok=True)

    # Counters for tracking
    stats = {
        'total_archives': len(urls),
        'downloaded_archives': 0,
        'skipped_archives': 0,
        'extracted_files': 0,
        'skipped_files': 0
    }

    for i, (url, filename) in enumerate(zip(urls, filenames), 1):
        archive_file = archive_path / filename

        # Skip if archive already exists
        if archive_file.exists():
            print(
                f"[{i}/{stats['total_archives']}] Archive already exists: {filename}"
            )
            stats['skipped_archives'] += 1
        else:
            # Download archive
            print(f"[{i}/{stats['total_archives']}] Downloading {filename}...")
            result = download_file(url, archive_file)

            if not result['success']:
                print(f"Failed to download {filename}: {result['message']}")
                continue

            stats['downloaded_archives'] += 1

        # Extract selective files
        extract_success, extracted_count, skipped_count, error = extract_selective_files_from_archive(
            archive_file, extracted_path, start_dt, end_dt, time_to_keep)

        if not extract_success:
            print(f"  Error extracting from {filename}: {error}")
            continue

        stats['extracted_files'] += extracted_count
        stats['skipped_files'] += skipped_count

        if skipped_count > 0:
            print(
                f"  Files extracted: {extracted_count}, skipped (already exist): {skipped_count}"
            )
        else:
            print(
                f"  Total files extracted from {filename}: {extracted_count}")

        if not keep_raw:
            archive_file.unlink()
            print(f"  Deleted archive: {filename}")

    print(f"\n=== SUMMARY ===")
    print(f"Total archives processed: {stats['total_archives']}")
    print(f"Archives downloaded: {stats['downloaded_archives']}")
    print(f"Archives skipped (already existed): {stats['skipped_archives']}")
    print(f"Files extracted: {stats['extracted_files']}")
    print(f"Files skipped (already existed): {stats['skipped_files']}")

    return stats
