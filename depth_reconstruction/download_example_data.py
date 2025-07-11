# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
# """Download and extract the SUN3D dataset for depth reconstruction."""
import hashlib
import logging
import os
import shutil
from pathlib import Path

import requests
import tqdm


def get_sha256(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        # Calculate the SHA256 hash of the file
        digest_object = hashlib.file_digest(f, "sha256")
        file_sha256 = digest_object.hexdigest()
    return file_sha256


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")

    current_path = Path(__file__).resolve().parent
    data_dir = current_path / "data" / "sun3d"

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip_filename = "sun3d-mit_76_studyroom-76-1studyroom2.zip"
    dataset_url = f"http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/{dataset_zip_filename}"
    dataset_sha256 = "5af09bc9e47a116ec31bed7e4a8712383bdab9eadf683cb8938ffd6448a00ede"

    download_path = data_dir / dataset_zip_filename

    do_download = True
    if os.path.exists(download_path):
        do_download = get_sha256(download_path) != dataset_sha256
        if do_download:
            logging.info("File exists but SHA256 does not match. Re-downloading.")
            os.remove(download_path)

    if not do_download:
        logging.info("File already exists and SHA256 matches. Skipping download.")
    else:
        logging.info(f"Downloading dataset from {dataset_url} to {download_path}")
        assert (
            not download_path.exists()
        ), f"File {download_path} already exists but SHA256 does not match. Please remove the file manually."

        response = requests.get(dataset_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            assert total_size > 0, "Downloaded file is empty."
            with open(download_path, "wb") as f:
                with tqdm.tqdm(total=total_size, unit="B", unit_scale=True, desc=dataset_zip_filename) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            logging.info("File downloaded successfully.")
            logging.info(f"SHA256 of downloaded file: {get_sha256(download_path)}")
            if get_sha256(download_path) != dataset_sha256:
                logging.error("SHA256 mismatch after download.")
                return
        else:
            logging.error(f"Failed to download file from url {dataset_url}. HTTP Status code: {response.status_code}")
            return

    logging.info(f"Extracting {download_path} to {data_dir}")
    shutil.unpack_archive(download_path, extract_dir=data_dir, format="zip")
    logging.info(f"Extraction complete. Data available at {data_dir}")


if __name__ == "__main__":
    main()
