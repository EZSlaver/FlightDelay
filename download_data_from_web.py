import requests
import os
import zipfile


def download_file(from_url, output_file_name, buffer_size = 8192):
    with requests.get(from_url, stream=True) as r:
        r.raise_for_status()
        file_size = float(r.headers['content-length'])
        download_prog_str = "\rDownload progress: %d/%d (%.2f%%)"
        progress = 0
        with open(output_file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=buffer_size):
                if chunk:  # filter out keep-alive new chunks
                    progress += len(chunk)
                    f.write(chunk)
                    print(download_prog_str % (progress, file_size, progress * 100. / file_size), end='')

        print()


if __name__ == "__main__":
    url_base = 'https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip'

    raw_data_folder = os.path.join('Data', 'All')

    start_month, start_year = 5, 2014
    # Non Inclusive
    end_month, end_year = 5, 2018

    downloaded = []
    while start_month != end_month or start_year != end_year:

        url = url_base.format(month=start_month, year=start_year)
        dest = os.path.join(raw_data_folder, '%d_%d.zip' % (start_month, start_year))
        print('Downloading file from "%s" to "%s"' % (url, dest))

        download_file(url, dest)

        print('Download completed. Extracting file.')

        with zipfile.ZipFile(dest, 'r') as zip_ref:
            target_file = None
            for name in zip_ref.namelist():
                if name.endswith('csv'):
                    target_file = name
                    break

            if target_file is None:
                print("WARNING: Zip '%s' doesn't contain CSV files. Continuing w/o deletion.")
                continue

            zip_ref.extract(target_file, raw_data_folder)
            os.rename(os.path.join(raw_data_folder, target_file),
                      os.path.join(raw_data_folder, "%02d_%02d.csv" % (start_year % 100, start_month)))

        print('Extraction finished.')

        os.remove(dest)

        start_month += 1
        if start_month == 13:
            start_month = 1
            start_year += 1

    print('Done.')
