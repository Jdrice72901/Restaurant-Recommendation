import argparse
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from recommender.config import load_config  # noqa: E402

INNER_ARCHIVE = 'Yelp JSON/yelp_dataset.tar'
DOCUMENTATION = 'Yelp JSON/Yelp Dataset Documentation & ToS copy.pdf'
AGREEMENT = 'Dataset_User_Agreement.pdf'

DATASET_URL = 'https://business.yelp.com/external-assets/files/Yelp-JSON.zip'
EXPECTED_FIELDS = {
    'business': {
        'business_id',
        'name',
        'categories',
        'latitude',
        'longitude',
        'attributes',
        'stars',
        'review_count',
        'is_open',
    },
    'review': {'review_id', 'user_id', 'business_id', 'stars', 'text', 'date'},
    'user': {'user_id', 'name', 'review_count'},
    'tip': {'user_id', 'business_id', 'text', 'date'},
    'checkin': {'business_id', 'date'},
}
EXPECTED_FILES = {
    f'yelp_academic_dataset_{kind}.json': fields for kind, fields in EXPECTED_FIELDS.items()
}


def download(url, destination):
    partial = destination.with_name(destination.name + '.part')
    print(f"Downloading {url}", flush=True)
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 120),
            headers={
                'Accept-Encoding': 'identity',
                'User-Agent': 'Mozilla/5.0',
                'Referer': 'https://business.yelp.com/data/resources/open-dataset/',
            },
        ) as response:
            response.raise_for_status()
            expected_size = response.headers.get('Content-Length')
            size = 0
            next_report = 256 * 1024 * 1024

            # Write in chunks for progress reporting and to avoid memory issues
            with partial.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
                    if size >= next_report:
                        print(f"  Downloaded {size / 1024**3:.2f} GiB", flush=True)
                        next_report += 256 * 1024 * 1024

            if size == 0 or (expected_size and size != int(expected_size)):
                raise ValueError(f"Incomplete: received {size} bytes; expected {expected_size}.")

        # Check format before promoting an HTTP error page or unexpected payload
        with zipfile.ZipFile(partial) as archive:
            validate_archive(archive)

        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def validate_jsonl(path, fields):
    count = 0
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)

            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError(f"{path.name}, line {line_number}: invalid JSON") from exc

            if not isinstance(record, dict) or not fields.issubset(record):
                raise ValueError(
                    f"{path.name}, line {line_number}: expected object with fields {sorted(fields)}"
                )

            count += 1

    if not count:
        raise ValueError(f"{path.name} is empty.")

    print(f"  Validated {path.name}: {count:,} records", flush=True)
    return count


def validate_archive(archive):
    for name in (INNER_ARCHIVE, DOCUMENTATION):
        member = archive.getinfo(name)
        if member.file_size == 0:
            raise ValueError(f"Empty archive member: {name}")


def extract_file(source, destination, size):
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"Existing {destination} differs in size; move aside and retry.")

        if destination.name in EXPECTED_FILES:
            validate_jsonl(destination, EXPECTED_FILES[destination.name])

        else:
            with destination.open('rb') as existing:
                if existing.read(5) != b'%PDF-':
                    raise ValueError(f"Invalid PDF: {destination}")

        print(f"  Keeping existing {destination.name}", flush=True)
        return

    print(f"  Extracting {destination.name}", flush=True)
    partial = destination.with_name(destination.name + '.part')

    try:
        with partial.open('wb') as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)

        if partial.stat().st_size != size:
            raise ValueError(f"Incomplete extraction: {destination.name}")

        if destination.name in EXPECTED_FILES:
            validate_jsonl(partial, EXPECTED_FILES[destination.name])

        else:
            with partial.open('rb') as pdf:
                if pdf.read(5) != b'%PDF-':
                    raise ValueError(f"Invalid PDF: {destination.name}")

        partial.replace(destination)

    finally:
        partial.unlink(missing_ok=True)


def extract_archive(archive_path, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected = set(EXPECTED_FILES) | {AGREEMENT}
    found = set()

    with zipfile.ZipFile(archive_path) as archive:
        validate_archive(archive)
        documentation = archive.getinfo(DOCUMENTATION)

        with archive.open(documentation) as source:
            extract_file(source, raw_dir / Path(DOCUMENTATION).name, documentation.file_size)

        with archive.open(INNER_ARCHIVE) as compressed:
            with tarfile.open(fileobj=compressed, mode='r|gz') as dataset:
                for member in dataset:
                    # Only fixed, inspected filenames become output paths
                    if member.name not in expected or not member.isfile():
                        continue

                    source = dataset.extractfile(member)

                    if source is None:
                        raise ValueError(f"Failed to extract {member.name} from TAR")

                    with source:
                        extract_file(source, raw_dir / member.name, member.size)

                    found.add(member.name)

            while compressed.read(1024 * 1024):
                pass

    if missing := expected - found:
        raise ValueError(f"Dataset archive is missing: {', '.join(sorted(missing))}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', help="Config path; relative paths resolve against the repository root"
    )
    parser.add_argument(
        '--archive',
        help="Use an existing Yelp ZIP instead of downloading; relative to the repository root",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else REPO_ROOT / 'config.yaml'

    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    cfg = load_config(config_path)
    raw_dir = cfg['data']['raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)

    for kind in ('business', 'review'):
        if cfg['data'][f'{kind}_file'] != f'yelp_academic_dataset_{kind}.json':
            raise ValueError(f"data.{kind}_file must match the published Yelp filename.")

    default_archive = raw_dir / 'Yelp-JSON.zip'
    archive_path = Path(args.archive) if args.archive else default_archive

    if not archive_path.is_absolute():
        archive_path = REPO_ROOT / archive_path

    if not archive_path.exists():
        if args.archive:
            raise FileNotFoundError(archive_path)
        download(DATASET_URL, archive_path)

    else:
        print(f"Using existing archive: {archive_path}", flush=True)

    extract_archive(archive_path, raw_dir)
    print(f"Source ingestion complete: {raw_dir}\nNext: python scripts/build_index.py", flush=True)


if __name__ == '__main__':
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as exc:
        raise SystemExit(f"Ingestion failed: {exc}") from exc
